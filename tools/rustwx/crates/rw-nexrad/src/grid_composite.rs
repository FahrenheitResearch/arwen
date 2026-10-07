//! `rw_nexrad grid-composite`: 2D reflectivity frames (a nowcast, or the
//! observed MRMS composite) onto a model grid as radar latent heating
//! windows, in the same `gpuwm-obs.radar-tten-ref.v1` format `grid-ref`
//! writes from Level II volumes.
//!
//! ```text
//! rw_nexrad grid-composite --grid GRID.json --frames FRAMES_ROOT
//!     (--profile-from OBS_WINDOW/ref.f32 | --profile-table LIB.json)
//!     --start TIME --window-minutes M --windows N --out WINDOWS_ROOT
//!     [--echo-dbz 15] [--clear-dbz 5] [--oracle] [--threads N]
//! ```
//!
//! The frames are the plug point: any 2D source writes
//! `gpuwm-obs.nowcast-frames.v1` (`FRAMES_ROOT/nowcast.json` plus one raw
//! float32 `[members, ny, nx]` file per valid time), and this stage reads
//! only that.  A new source is new files, not new code here.
//!
//! **Map.**  Every model mass column takes its latitude and longitude from
//! the grid descriptor (the WPS projection, `grid_ref::ModelGrid`) and is
//! placed on the nearest cell of the source lattice through the lattice's
//! OWN projection and earth radius.  The two grids are never assumed to be
//! the same lattice because both say "3 km": StormScope's crop uses HRRR's
//! 6,371,229 m sphere and the WPS grid 6,370,000 m.  A lattice whose stated
//! corners do not reproduce from its projection numbers within
//! [`CORNER_TOLERANCE_M`] is refused.
//!
//! **Reduce.**  Over the members, a probability-matched mean on the covered
//! columns (NaN in at most half the members): the ensemble-mean field gives
//! the ranks, the pooled member values give the amplitudes.  One member is
//! taken as it is.
//!
//! **Classify.**  Echo where the reduced value is at least `--echo-dbz`;
//! observed clear (the whole column -99, so NOAA's rule withholds the
//! model's heating) where every member is finite and below `--clear-dbz`;
//! no coverage (-99999, the model's microphysics stands) everywhere else,
//! including NaN in more than half the members.
//!
//! **Expand.**  A composite has no vertical structure, so each echo column
//! takes the shape of a same-day profile library built from the observed
//! 3D window at the start time (lane 6's Level II window on the same grid,
//! which is causal): the median of (Z minus the column composite) and the
//! echo fraction per (composite bin, height above ground) cell.  A library
//! window that ends after `--start` is refused without `--oracle`: the
//! shapes would carry observations the forecast could not have had.  A level
//! is echo where the echo fraction is at least [`ECHO_FRACTION_MIN`]; the
//! column is then shifted so its maximum is the composite exactly.  Levels
//! the library does not put echo at are left to the model (no coverage),
//! never suppressed.
//!
//! **Determinism.**  Columns are reduced and expanded in parallel into
//! disjoint slots; the ranking sort and every median use total orders with
//! a fixed tie-break.  The output bytes do not depend on the thread count.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::{DateTime, Duration, NaiveDateTime, TimeZone, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::grid_ref::{
    round3, write_atomic, ModelGrid, NO_COVERAGE, OBSERVED_NO_ECHO, REF_SCHEMA,
};
use crate::s3::{boxed_error, hex_sha256, iso8601};

/// The frames receipt any 2D reflectivity source writes.
pub const FRAMES_SCHEMA: &str = "gpuwm-obs.nowcast-frames.v1";
/// The profile library this stage writes beside its windows.
pub const LIBRARY_SCHEMA: &str = "gpuwm-obs.radar-profile-library.v1";
/// The informational index of a windows root (`tools/radar_tten_windows.py`
/// writes the same schema for Level II windows).
pub const WINDOWS_SCHEMA: &str = "gpuwm-obs.radar-tten-windows.v1";
/// This subcommand's stdout record.
pub const COMPOSITE_SCHEMA: &str = "gpuwm-obs.radar-tten-composite.v1";
/// A frame file's dtype as the frames contract spells it (`<f4` is read
/// as the same thing).
pub const FRAME_DTYPE: &str = "float32-le";

pub const DEFAULT_ECHO_DBZ: f32 = 15.0;
pub const DEFAULT_CLEAR_DBZ: f32 = 5.0;
/// An observed frame may sit this far from a window end (MRMS files carry
/// off-cadence seconds); a nowcast frame must sit on it.
pub const OBSERVED_TOLERANCE_S: i64 = 120;
/// A lattice's stated corners must reproduce from its projection within this.
pub const CORNER_TOLERANCE_M: f64 = 100.0;
/// NOAA's echo floor (`radar_ref2tten.f90:182`): below it a covered point
/// is read as observed clear.  An expanded level that would land below it
/// is handed to the model instead, so a weak level is never read as clear.
pub const NOAA_ECHO_FLOOR_DBZ: f32 = 0.001;

/// Library composite bins: 20 dBZ upward in 5 dBZ steps, the last open.
pub const LIBRARY_MIN_DBZ: f64 = 20.0;
pub const LIBRARY_BIN_DBZ: f64 = 5.0;
pub const LIBRARY_BINS: usize = 10;
/// Library height bins above ground: 250 m steps to 18 km.
pub const HEIGHT_BIN_M: f64 = 250.0;
pub const HEIGHT_BINS: usize = 72;
/// A library cell needs this many observed gates to stand on its own.
pub const MIN_CELL_SAMPLES: u64 = 30;
/// A library built from fewer columns at [`LIBRARY_COUNT_DBZ`] or more is
/// refused unless a `--profile-table` is given.
pub const MIN_LIBRARY_COLUMNS: u64 = 500;
pub const LIBRARY_COUNT_DBZ: f64 = 30.0;
/// A level is expanded as echo where the library's echo fraction reaches this.
pub const ECHO_FRACTION_MIN: f64 = 0.5;

const DEG: f64 = std::f64::consts::PI / 180.0;

// ---------------------------------------------------------------------------
// time
// ---------------------------------------------------------------------------

/// The time spellings this stage reads: `2026-10-01T18:00:00Z`,
/// `2026-10-01T18:00Z`, and the compact `20261001T180000`.
pub fn parse_when(value: &str) -> Result<DateTime<Utc>, String> {
    let trimmed = value.trim();
    for format in [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%MZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y%m%dT%H%M%S",
        "%Y%m%dT%H%MZ",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Ok(Utc.from_utc_datetime(&naive));
        }
    }
    Err(format!(
        "unparseable time {value:?}: expected 2026-10-01T18:00:00Z, 2026-10-01T18:00Z or 20261001T180000"
    ))
}

/// The window directory name: the window END, UTC, minute resolution, as
/// `gpuwm.da.radar_tten.window_paths` forms it.
pub fn window_stamp(end: DateTime<Utc>) -> String {
    end.format("%Y%m%dT%H%MZ").to_string()
}

// ---------------------------------------------------------------------------
// the source lattice
// ---------------------------------------------------------------------------

/// A spherical Lambert conformal lattice in its own projection coordinates,
/// as `docs/nowcast-frames.md` defines it.
///
/// Projection coordinates are metres from the projection origin at
/// (`ref_lat`, `stand_lon`): with cone constant n,
/// `x = rho sin(n (lon - stand_lon))`, `y = rho0 - rho cos(n (lon - stand_lon))`
/// (pyproj's `+proj=lcc +lat_1 +lat_2 +lat_0=ref_lat +lon_0=stand_lon
/// +R=earth_radius_m`).  Cell (j, i) centre is at `(x0 + i dx, y0 + j dy)`,
/// and `dx`, `dy` are signed: a lattice stored north row first has `dy < 0`.
#[derive(Debug, Clone)]
pub struct LambertLattice {
    pub n: f64,
    pub f: f64,
    pub radius: f64,
    pub stand_lon: f64,
    ref_raw: (f64, f64),
    pub x0: f64,
    pub y0: f64,
    pub dx: f64,
    pub dy: f64,
    pub nx: usize,
    pub ny: usize,
}

/// Longitude difference folded into [-180, 180).
fn lon_delta(lon: f64, origin: f64) -> f64 {
    (lon - origin + 180.0).rem_euclid(360.0) - 180.0
}

impl LambertLattice {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        truelat1: f64,
        truelat2: f64,
        stand_lon: f64,
        ref_lat: f64,
        ref_lon: f64,
        earth_radius_m: f64,
        (nx, ny): (usize, usize),
        (dx, dy): (f64, f64),
        (x0, y0): (f64, f64),
    ) -> Result<Self, String> {
        for (name, value) in [
            ("truelat1", truelat1),
            ("truelat2", truelat2),
            ("stand_lon", stand_lon),
            ("ref_lat", ref_lat),
            ("ref_lon", ref_lon),
            ("x0_m", x0),
            ("y0_m", y0),
        ] {
            if !value.is_finite() {
                return Err(format!("lambert lattice {name} is not finite"));
            }
        }
        if !(earth_radius_m.is_finite() && earth_radius_m > 1.0e6) {
            return Err(format!("lambert lattice earth_radius_m {earth_radius_m} is not a sphere"));
        }
        if !(dx.is_finite() && dx != 0.0 && dy.is_finite() && dy != 0.0) {
            return Err(format!("lambert lattice spacing dx={dx} dy={dy} must be finite and nonzero"));
        }
        if lon_delta(ref_lon, stand_lon).abs() > 1.0e-9 {
            return Err(format!(
                "lambert lattice ref_lon {ref_lon} is not stand_lon {stand_lon}: the frames \
                 contract puts the projection origin at (ref_lat, stand_lon), and a lattice \
                 that names another meridian is ambiguous about where its cells are"
            ));
        }
        if nx == 0 || ny == 0 {
            return Err(format!("lambert lattice has an empty axis: nx={nx} ny={ny}"));
        }
        let (p1, p2) = (truelat1 * DEG, truelat2 * DEG);
        let t = |p: f64| (std::f64::consts::FRAC_PI_4 + p / 2.0).tan();
        let n = if (truelat1 - truelat2).abs() < 1.0e-9 {
            p1.sin()
        } else {
            (p1.cos() / p2.cos()).ln() / (t(p2) / t(p1)).ln()
        };
        if !n.is_finite() || n.abs() < 1.0e-6 {
            return Err(format!(
                "lambert lattice truelats {truelat1}, {truelat2} give a degenerate cone ({n})"
            ));
        }
        let f = p1.cos() * t(p1).powf(n) / n;
        let mut lattice = Self {
            n,
            f,
            radius: earth_radius_m,
            stand_lon,
            ref_raw: (0.0, 0.0),
            x0,
            y0,
            dx,
            dy,
            nx,
            ny,
        };
        lattice.ref_raw = lattice.raw(ref_lat, ref_lon);
        Ok(lattice)
    }

    fn raw(&self, lat: f64, lon: f64) -> (f64, f64) {
        let rho = self.radius * self.f
            / (std::f64::consts::FRAC_PI_4 + lat * DEG / 2.0).tan().powf(self.n);
        let theta = self.n * lon_delta(lon, self.stand_lon) * DEG;
        (rho * theta.sin(), -rho * theta.cos())
    }

    /// Projection coordinates, metres from the reference point.
    pub fn project(&self, lat: f64, lon: f64) -> (f64, f64) {
        let (x, y) = self.raw(lat, lon);
        (x - self.ref_raw.0, y - self.ref_raw.1)
    }

    /// The inverse of [`Self::project`].
    pub fn unproject(&self, x: f64, y: f64) -> (f64, f64) {
        let (xr, yr) = (x + self.ref_raw.0, y + self.ref_raw.1);
        let sign = self.n.signum();
        let rho = sign * (xr * xr + yr * yr).sqrt();
        let theta = (sign * xr).atan2(-sign * yr);
        let lat = 2.0 * (self.radius * self.f / rho).powf(1.0 / self.n).atan()
            - std::f64::consts::FRAC_PI_2;
        let lon = self.stand_lon + theta / self.n / DEG;
        (lat / DEG, (lon + 180.0).rem_euclid(360.0) - 180.0)
    }
}

/// A regular latitude/longitude lattice: cell (j, i) centre at
/// `(lat0 + j dlat, lon0 + i dlon)`.
#[derive(Debug, Clone)]
pub struct LatLonLattice {
    pub lat0: f64,
    pub lon0: f64,
    pub dlat: f64,
    pub dlon: f64,
    pub nx: usize,
    pub ny: usize,
}

#[derive(Debug, Clone)]
pub enum Lattice {
    Lambert(LambertLattice),
    LatLon(LatLonLattice),
}

impl Lattice {
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Lattice::Lambert(l) => (l.ny, l.nx),
            Lattice::LatLon(l) => (l.ny, l.nx),
        }
    }

    /// Zero-based fractional (i, j) of a point.
    pub fn fractional_index(&self, lat: f64, lon: f64) -> (f64, f64) {
        match self {
            Lattice::Lambert(l) => {
                let (x, y) = l.project(lat, lon);
                ((x - l.x0) / l.dx, (y - l.y0) / l.dy)
            }
            Lattice::LatLon(l) => {
                let di = (lon - l.lon0).rem_euclid(360.0);
                // A point just west of lon0 folds to ~360; keep it negative so
                // it falls outside rather than onto the far edge.
                let di = if di > 180.0 + (l.nx as f64 * l.dlon).min(180.0) { di - 360.0 } else { di };
                (di / l.dlon, (lat - l.lat0) / l.dlat)
            }
        }
    }

    /// The nearest cell (row-major index), or `None` outside the lattice.
    pub fn nearest(&self, lat: f64, lon: f64) -> Option<usize> {
        let (ny, nx) = self.shape();
        let (fi, fj) = self.fractional_index(lat, lon);
        let i = fi.round_ties_even();
        let j = fj.round_ties_even();
        if i >= 0.0 && i < nx as f64 && j >= 0.0 && j < ny as f64 {
            Some(j as usize * nx + i as usize)
        } else {
            None
        }
    }

    /// Cell (j, i) centre as (lat, lon).
    pub fn cell_latlon(&self, j: usize, i: usize) -> (f64, f64) {
        match self {
            Lattice::Lambert(l) => l.unproject(l.x0 + i as f64 * l.dx, l.y0 + j as f64 * l.dy),
            Lattice::LatLon(l) => {
                let lon = l.lon0 + i as f64 * l.dlon;
                (l.lat0 + j as f64 * l.dlat, (lon + 180.0).rem_euclid(360.0) - 180.0)
            }
        }
    }

    pub fn earth_radius_m(&self) -> f64 {
        match self {
            Lattice::Lambert(l) => l.radius,
            Lattice::LatLon(_) => crate::grid_ref::EARTH_RADIUS_M,
        }
    }

    /// The four corner cell centres under the contract's keys: `j0_i0`,
    /// `j0_in` (row 0, last column), `jn_i0` (last row, column 0), `jn_in`.
    pub fn corners(&self) -> [(&'static str, (f64, f64)); 4] {
        let (ny, nx) = self.shape();
        [
            ("j0_i0", self.cell_latlon(0, 0)),
            ("j0_in", self.cell_latlon(0, nx - 1)),
            ("jn_i0", self.cell_latlon(ny - 1, 0)),
            ("jn_in", self.cell_latlon(ny - 1, nx - 1)),
        ]
    }

    /// The `corners_latlon` block this lattice states, longitude in [-180, 180).
    pub fn corners_json(&self) -> serde_json::Value {
        let mut block = serde_json::Map::new();
        for (key, (lat, lon)) in self.corners() {
            block.insert(key.to_string(), serde_json::json!([lat, lon]));
        }
        serde_json::Value::Object(block)
    }

    /// Parse a frames receipt's `lattice` block and prove its corners.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        let kind = value
            .get("kind")
            .and_then(|v| v.as_str())
            .ok_or("frames lattice has no kind")?;
        let num = |name: &str| -> Result<f64, String> {
            value
                .get(name)
                .and_then(|v| v.as_f64())
                .ok_or_else(|| format!("frames lattice ({kind}) has no number {name}"))
        };
        let count = |name: &str| -> Result<usize, String> {
            value
                .get(name)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| format!("frames lattice ({kind}) has no count {name}"))
        };
        let row_order = |positive: bool, axis: &str| -> Result<(), String> {
            let Some(order) = value.get("row_order") else {
                return if kind == "latlon" {
                    Err("latlon lattice has no row_order".to_string())
                } else {
                    Ok(())
                };
            };
            let declared_south_first = match order.as_str() {
                Some("south_to_north") => true,
                Some("north_to_south") => false,
                other => {
                    return Err(format!(
                        "{kind} lattice row_order {other:?} is neither south_to_north nor north_to_south"
                    ))
                }
            };
            if declared_south_first != positive {
                return Err(format!(
                    "{kind} lattice row_order {order} disagrees with the sign of {axis}: rows \
                     would be mirrored and every column heated from the other side of the domain"
                ));
            }
            Ok(())
        };
        let lattice = match kind {
            "lambert" => {
                let l = LambertLattice::new(
                    num("truelat1")?,
                    num("truelat2")?,
                    num("stand_lon")?,
                    num("ref_lat")?,
                    num("ref_lon")?,
                    num("earth_radius_m")?,
                    (count("nx")?, count("ny")?),
                    (num("dx_m")?, num("dy_m")?),
                    (num("x0_m")?, num("y0_m")?),
                )?;
                row_order(l.dy > 0.0, "dy_m")?;
                Lattice::Lambert(l)
            }
            "latlon" => {
                let l = LatLonLattice {
                    lat0: num("lat0")?,
                    lon0: num("lon0")?,
                    dlat: num("dlat")?,
                    dlon: num("dlon")?,
                    nx: count("nx")?,
                    ny: count("ny")?,
                };
                if l.nx == 0 || l.ny == 0 {
                    return Err(format!("latlon lattice has an empty axis: nx={} ny={}", l.nx, l.ny));
                }
                if !(l.dlon.is_finite() && l.dlon > 0.0) {
                    return Err(format!("latlon lattice dlon {} must be positive (east)", l.dlon));
                }
                if !(l.dlat.is_finite() && l.dlat != 0.0) {
                    return Err(format!("latlon lattice dlat {} must be nonzero", l.dlat));
                }
                row_order(l.dlat > 0.0, "dlat")?;
                Lattice::LatLon(l)
            }
            other => return Err(format!("frames lattice kind {other:?} is neither lambert nor latlon")),
        };
        lattice.check_corners(value.get("corners_latlon"))?;
        Ok(lattice)
    }

    /// Refuse a lattice whose stated corners do not reproduce.
    pub fn check_corners(&self, stated: Option<&serde_json::Value>) -> Result<(), String> {
        let block = stated.and_then(|v| v.as_object()).ok_or(
            "frames lattice states no corners_latlon {j0_i0, j0_in, jn_i0, jn_in}; a lattice \
             that cannot be checked is not used",
        )?;
        let radius = self.earth_radius_m();
        for (key, (lat, lon)) in self.corners() {
            let pair = block
                .get(key)
                .and_then(|p| p.as_array())
                .filter(|p| p.len() == 2)
                .ok_or_else(|| format!("corners_latlon.{key} is not a [lat, lon] pair"))?;
            let (slat, slon) = match (pair[0].as_f64(), pair[1].as_f64()) {
                (Some(a), Some(b)) => (a, b),
                _ => return Err(format!("corners_latlon.{key} is not numeric")),
            };
            let distance = haversine_m(lat, lon, slat, slon, radius);
            if !(distance <= CORNER_TOLERANCE_M) {
                return Err(format!(
                    "lattice corner {key} is stated at ({slat}, {slon}) but its projection \
                     numbers put it at ({lat:.6}, {lon:.6}), {distance:.0} m away (limit \
                     {CORNER_TOLERANCE_M} m): heating would land in the wrong columns"
                ));
            }
        }
        Ok(())
    }
}

pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64, radius: f64) -> f64 {
    let (p1, p2) = (lat1 * DEG, lat2 * DEG);
    let dp = p2 - p1;
    let dl = lon_delta(lon2, lon1) * DEG;
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * radius * a.clamp(0.0, 1.0).sqrt().asin()
}

/// Every model column's source cell, `None` outside the lattice.
pub fn map_columns(grid: &ModelGrid, lattice: &Lattice) -> Vec<Option<u32>> {
    (0..grid.nx * grid.ny)
        .into_par_iter()
        .map(|cell| {
            let (lat, lon) = grid.cell_latlon(cell);
            lattice.nearest(lat, lon).map(|s| s as u32)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// the frames receipt
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct FrameEntry {
    pub valid: String,
    #[serde(default)]
    pub lead_minutes: Option<f64>,
    pub file: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub sha256: String,
    /// An observed frame's own object time, when it differs from `valid`.
    #[serde(default)]
    pub stamp: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Frames {
    pub root: PathBuf,
    pub receipt_sha256: String,
    pub kind: String,
    pub id: String,
    pub issue_time: DateTime<Utc>,
    pub causal: bool,
    pub members: usize,
    pub lattice: Lattice,
    pub frames: Vec<(DateTime<Utc>, FrameEntry)>,
}

impl Frames {
    pub fn load(root: &Path) -> Result<Self, String> {
        let path = root.join("nowcast.json");
        let bytes = std::fs::read(&path)
            .map_err(|e| format!("cannot read frames receipt {}: {e}", path.display()))?;
        let doc: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("frames receipt {} is not JSON: {e}", path.display()))?;
        let text = |name: &str| doc.get(name).and_then(|v| v.as_str());
        if text("schema") != Some(FRAMES_SCHEMA) || text("status") != Some("READY") {
            return Err(format!(
                "{} declares schema {:?} status {:?}; expected {FRAMES_SCHEMA:?} READY",
                path.display(),
                text("schema"),
                text("status")
            ));
        }
        if text("variable") != Some("refc") {
            return Err(format!(
                "{} carries variable {:?}; this stage reads the composite `refc` only",
                path.display(),
                text("variable")
            ));
        }
        let source = doc.get("source").ok_or("frames receipt has no source block")?;
        let kind = source
            .get("kind")
            .and_then(|v| v.as_str())
            .ok_or("frames source has no kind")?
            .to_string();
        if kind != "nowcast" && kind != "observed" {
            return Err(format!("frames source kind {kind:?} is neither nowcast nor observed"));
        }
        let id = source
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("frames source has no id")?
            .to_string();
        let issue_time = parse_when(text("issue_time").ok_or("frames receipt has no issue_time")?)?;
        let causal = doc
            .get("causal")
            .and_then(|v| v.as_bool())
            .ok_or("frames receipt states no causal flag")?;
        let members = doc
            .get("members")
            .and_then(|v| v.as_u64())
            .filter(|m| *m >= 1)
            .ok_or("frames receipt states no positive member count")? as usize;
        let lattice = Lattice::from_json(doc.get("lattice").ok_or("frames receipt has no lattice")?)?;
        let entries: Vec<FrameEntry> = serde_json::from_value(
            doc.get("frames").cloned().ok_or("frames receipt lists no frames")?,
        )
        .map_err(|e| format!("frames receipt frames[] is malformed: {e}"))?;
        let (ny, nx) = lattice.shape();
        let mut frames = Vec::with_capacity(entries.len());
        for entry in entries {
            if entry.shape != [members, ny, nx] {
                return Err(format!(
                    "frame {} has shape {:?}, but the receipt states {members} members on a \
                     {ny} x {nx} lattice: the members' shapes disagree and heating would land \
                     in the wrong columns",
                    entry.valid, entry.shape
                ));
            }
            if entry.dtype != FRAME_DTYPE && entry.dtype != "<f4" {
                return Err(format!(
                    "frame {} has dtype {:?}, expected {FRAME_DTYPE}",
                    entry.valid, entry.dtype
                ));
            }
            frames.push((parse_when(&entry.valid)?, entry));
        }
        if frames.is_empty() {
            return Err("frames receipt lists no frames".to_string());
        }
        Ok(Self {
            root: root.to_path_buf(),
            receipt_sha256: hex_sha256(&bytes),
            kind,
            id,
            issue_time,
            causal,
            members,
            lattice,
            frames,
        })
    }

    /// The frame valid at `end`: exactly, or for observed frames the nearest
    /// within [`OBSERVED_TOLERANCE_S`].
    pub fn frame_at(&self, end: DateTime<Utc>) -> Result<&(DateTime<Utc>, FrameEntry), String> {
        let tolerance = if self.kind == "observed" { OBSERVED_TOLERANCE_S } else { 0 };
        self.frames
            .iter()
            .filter(|(valid, _)| (*valid - end).num_seconds().abs() <= tolerance)
            .min_by_key(|(valid, _)| ((*valid - end).num_seconds().abs(), *valid))
            .ok_or_else(|| {
                format!(
                    "no frame is valid at window end {} (tolerance {tolerance} s): part of the \
                     forced period would run on the model while the record says forced",
                    iso8601(end)
                )
            })
    }

    /// One frame's members as float32, after proving its digest.
    pub fn read(&self, entry: &FrameEntry) -> Result<Vec<f32>, String> {
        let named = PathBuf::from(&entry.file);
        let path = if named.is_absolute() { named } else { self.root.join(named) };
        let bytes = std::fs::read(&path).map_err(|e| format!("cannot read frame {}: {e}", path.display()))?;
        let expected = entry.shape.iter().product::<usize>() * 4;
        if bytes.len() != expected {
            return Err(format!(
                "frame {} holds {} bytes, its shape {:?} needs {expected}",
                path.display(),
                bytes.len(),
                entry.shape
            ));
        }
        let digest = hex_sha256(&bytes);
        if digest != entry.sha256 {
            return Err(format!(
                "frame {} has sha256 {digest}, the receipt states {}",
                path.display(),
                entry.sha256
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// member reduction and classification
// ---------------------------------------------------------------------------

/// What a model column is after the members are reduced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnClass {
    Echo(f32),
    Clear,
    NoCoverage,
}

/// Reduce and classify every model column of one frame.
///
/// `members` holds `[members, source cells]` float32 values and `mapping`
/// each model column's source cell.
pub fn reduce_columns(
    members: &[f32],
    member_count: usize,
    mapping: &[Option<u32>],
    echo_dbz: f32,
    clear_dbz: f32,
) -> Vec<ColumnClass> {
    let cells = members.len() / member_count;
    // Per column: finite member count, mean over finite members, and
    // whether every member is finite and clear.
    let stats: Vec<Option<(usize, f64, bool)>> = mapping
        .par_iter()
        .map(|source| {
            source.map(|s| {
                let s = s as usize;
                let mut finite = 0usize;
                let mut sum = 0.0f64;
                let mut all_clear = true;
                for m in 0..member_count {
                    let v = members[m * cells + s];
                    if v.is_finite() {
                        finite += 1;
                        sum += v as f64;
                        if !(v < clear_dbz) {
                            all_clear = false;
                        }
                    } else {
                        all_clear = false;
                    }
                }
                let mean = if finite > 0 { sum / finite as f64 } else { f64::NAN };
                (finite, mean, all_clear && finite == member_count)
            })
        })
        .collect();
    let covered = |finite: usize| finite > 0 && finite * 2 >= member_count;
    let mut reduced: Vec<f32> = vec![f32::NAN; mapping.len()];
    if member_count == 1 {
        reduced
            .par_iter_mut()
            .zip(mapping.par_iter())
            .for_each(|(slot, source)| {
                if let Some(s) = source {
                    *slot = members[*s as usize];
                }
            });
    } else {
        let ranked: Vec<usize> = {
            let mut order: Vec<usize> = (0..mapping.len())
                .filter(|&c| matches!(stats[c], Some((finite, _, _)) if covered(finite)))
                .collect();
            order.par_sort_by(|&a, &b| {
                let (ma, mb) = (stats[a].unwrap().1, stats[b].unwrap().1);
                mb.total_cmp(&ma).then(a.cmp(&b))
            });
            order
        };
        let mut pool: Vec<f32> = ranked
            .iter()
            .flat_map(|&c| {
                let s = mapping[c].unwrap() as usize;
                (0..member_count).map(move |m| members[m * cells + s])
            })
            .filter(|v| v.is_finite())
            .collect();
        pool.par_sort_by(|a, b| b.total_cmp(a));
        // Each ranked column owns as many pooled values as it has finite
        // members, in rank order, and takes the middle one of its share.
        // With every member finite this is the textbook PMM (every M-th
        // value); a column with a missing member does not shift the share
        // of every column ranked below it.
        let mut owned = 0usize;
        for &column in &ranked {
            let finite = stats[column].unwrap().0;
            reduced[column] = pool[(owned + finite / 2).min(pool.len() - 1)];
            owned += finite;
        }
    }
    (0..mapping.len())
        .into_par_iter()
        .map(|c| match stats[c] {
            None => ColumnClass::NoCoverage,
            Some((_, _, true)) => ColumnClass::Clear,
            Some((finite, _, false)) => {
                let v = reduced[c];
                if covered(finite) && v.is_finite() && v >= echo_dbz {
                    ColumnClass::Echo(v)
                } else {
                    ColumnClass::NoCoverage
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// the profile library
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LibraryCell {
    /// Median of (Z minus column composite) over echo gates, dB.
    pub offset_db: Option<f64>,
    /// Echo gates over observed gates (echo + observed clear).
    pub echo_fraction: f64,
    pub echo: u64,
    pub clear: u64,
    /// The composite bin this cell's shape was taken from, when it was too
    /// thin to stand on its own.
    pub filled_from: Option<usize>,
}

impl LibraryCell {
    pub fn samples(&self) -> u64 {
        self.echo + self.clear
    }

    fn usable(&self) -> Option<f64> {
        let populated = self.samples() >= MIN_CELL_SAMPLES || self.filled_from.is_some();
        match self.offset_db {
            Some(offset) if populated && self.echo_fraction >= ECHO_FRACTION_MIN => Some(offset),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileLibrary {
    pub schema: String,
    pub status: String,
    pub composite_bins_dbz: Vec<f64>,
    pub height_bin_m: f64,
    pub height_bins: usize,
    pub min_samples: u64,
    pub echo_fraction_min: f64,
    pub columns: LibraryColumns,
    /// `[composite bin][height bin]`.
    pub cells: Vec<Vec<LibraryCell>>,
    pub source: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LibraryColumns {
    /// Columns at or above the first composite bin.
    pub binned: u64,
    /// Columns at or above [`LIBRARY_COUNT_DBZ`].
    pub at_or_above_30_dbz: u64,
}

/// The composite bin a column composite falls in; echo weaker than the
/// first bin takes the first bin's shape.
pub fn composite_bin(composite: f64) -> usize {
    if composite < LIBRARY_MIN_DBZ {
        return 0;
    }
    (((composite - LIBRARY_MIN_DBZ) / LIBRARY_BIN_DBZ).floor() as usize).min(LIBRARY_BINS - 1)
}

/// Height bin of a level's centre above the column's ground.
pub fn height_bin(agl_m: f64) -> Option<usize> {
    if !(agl_m >= 0.0) {
        return None;
    }
    let bin = (agl_m / HEIGHT_BIN_M).floor() as usize;
    (bin < HEIGHT_BINS).then_some(bin)
}

/// Each column's level height bins, `[column][level]`.
pub fn level_height_bins(grid: &ModelGrid) -> Vec<Vec<Option<usize>>> {
    (0..grid.nx * grid.ny)
        .into_par_iter()
        .map(|cell| {
            let z = grid.column_z_w(cell % grid.nx, cell / grid.nx);
            (0..grid.nz)
                .map(|k| height_bin(0.5 * (z[k] + z[k + 1]) - z[0]))
                .collect()
        })
        .collect()
}

fn median(values: &mut [f32]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[mid] as f64
    } else {
        (values[mid - 1] as f64 + values[mid] as f64) / 2.0
    })
}

/// Build the library from an observed 3D window `(nz, ny, nx)` in NOAA's
/// convention, on the grid whose level bins are given.
pub fn build_library(
    window: &[f32],
    nz: usize,
    columns: usize,
    bins: &[Vec<Option<usize>>],
    source: serde_json::Value,
) -> ProfileLibrary {
    // Per column: its composite bin and (height bin, offset or clear) samples.
    type Samples = Option<(usize, bool, Vec<(usize, Option<f32>)>)>;
    let per_column: Vec<Samples> = (0..columns)
        .into_par_iter()
        .map(|c| {
            let mut composite = f32::NEG_INFINITY;
            for k in 0..nz {
                let v = window[k * columns + c];
                if v >= NOAA_ECHO_FLOOR_DBZ && v > composite {
                    composite = v;
                }
            }
            if !(composite as f64 >= LIBRARY_MIN_DBZ) {
                return None;
            }
            let mut samples = Vec::new();
            for (k, bin) in bins[c].iter().enumerate() {
                let Some(h) = bin else { continue };
                let v = window[k * columns + c];
                if v >= NOAA_ECHO_FLOOR_DBZ {
                    samples.push((*h, Some(v - composite)));
                } else if v > -100.0 {
                    samples.push((*h, None));
                }
            }
            Some((composite_bin(composite as f64), composite as f64 >= LIBRARY_COUNT_DBZ, samples))
        })
        .collect();
    let mut offsets: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); HEIGHT_BINS]; LIBRARY_BINS];
    let mut cells: Vec<Vec<LibraryCell>> = vec![vec![LibraryCell::default(); HEIGHT_BINS]; LIBRARY_BINS];
    let mut counted = LibraryColumns::default();
    for column in per_column.into_iter().flatten() {
        let (b, strong, samples) = column;
        counted.binned += 1;
        if strong {
            counted.at_or_above_30_dbz += 1;
        }
        for (h, sample) in samples {
            match sample {
                Some(offset) => {
                    cells[b][h].echo += 1;
                    offsets[b][h].push(offset);
                }
                None => cells[b][h].clear += 1,
            }
        }
    }
    for b in 0..LIBRARY_BINS {
        for h in 0..HEIGHT_BINS {
            let cell = &mut cells[b][h];
            cell.offset_db = median(&mut offsets[b][h]);
            let samples = cell.samples();
            cell.echo_fraction = if samples > 0 { cell.echo as f64 / samples as f64 } else { 0.0 };
        }
    }
    // A thin cell takes the nearest populated composite bin at its height.
    let own = cells.clone();
    for b in 0..LIBRARY_BINS {
        for h in 0..HEIGHT_BINS {
            if own[b][h].samples() >= MIN_CELL_SAMPLES {
                continue;
            }
            let donor = (1..LIBRARY_BINS).find_map(|d| {
                [b.checked_sub(d), Some(b + d).filter(|v| *v < LIBRARY_BINS)]
                    .into_iter()
                    .flatten()
                    .find(|&other| own[other][h].samples() >= MIN_CELL_SAMPLES)
            });
            if let Some(other) = donor {
                let cell = &mut cells[b][h];
                cell.offset_db = own[other][h].offset_db;
                cell.echo_fraction = own[other][h].echo_fraction;
                cell.filled_from = Some(other);
            }
        }
    }
    ProfileLibrary {
        schema: LIBRARY_SCHEMA.to_string(),
        status: "READY".to_string(),
        composite_bins_dbz: (0..LIBRARY_BINS)
            .map(|b| LIBRARY_MIN_DBZ + b as f64 * LIBRARY_BIN_DBZ)
            .collect(),
        height_bin_m: HEIGHT_BIN_M,
        height_bins: HEIGHT_BINS,
        min_samples: MIN_CELL_SAMPLES,
        echo_fraction_min: ECHO_FRACTION_MIN,
        columns: counted,
        cells,
        source,
    }
}

impl ProfileLibrary {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read --profile-table {}: {e}", path.display()))?;
        let library: ProfileLibrary = serde_json::from_str(&text)
            .map_err(|e| format!("--profile-table {} is not a {LIBRARY_SCHEMA} document: {e}", path.display()))?;
        if library.schema != LIBRARY_SCHEMA || library.status != "READY" {
            return Err(format!(
                "--profile-table {} declares {:?} {:?}; expected {LIBRARY_SCHEMA:?} READY",
                path.display(),
                library.schema,
                library.status
            ));
        }
        if library.cells.len() != LIBRARY_BINS
            || library.cells.iter().any(|row| row.len() != HEIGHT_BINS)
            || library.height_bin_m != HEIGHT_BIN_M
        {
            return Err(format!(
                "--profile-table {} is not binned {LIBRARY_BINS} x {HEIGHT_BINS} at {HEIGHT_BIN_M} m",
                path.display()
            ));
        }
        Ok(library)
    }

    /// One echo column expanded onto its levels: dBZ where the library puts
    /// echo, no coverage elsewhere, shifted so the maximum is `composite`.
    pub fn expand(&self, composite: f32, levels: &[Option<usize>], out: &mut [f32]) -> usize {
        let b = composite_bin(composite as f64);
        let offsets: Vec<Option<f64>> = levels
            .iter()
            .map(|h| h.and_then(|h| self.cells[b][h].usable()))
            .collect();
        let top = offsets.iter().flatten().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut weak = 0;
        for (slot, offset) in out.iter_mut().zip(&offsets) {
            *slot = match offset {
                Some(offset) => {
                    let z = (composite as f64 + (offset - top)) as f32;
                    if z >= NOAA_ECHO_FLOOR_DBZ {
                        z
                    } else {
                        weak += 1;
                        NO_COVERAGE
                    }
                }
                None => NO_COVERAGE,
            };
        }
        weak
    }
}

// ---------------------------------------------------------------------------
// the observed window the library is built from
// ---------------------------------------------------------------------------

/// Read a `gpuwm-obs.radar-tten-ref.v1` window (its `ref.json` beside the
/// data) and hold it to the grid: identity, shape, byte count and digest.
pub fn read_window(data: &Path, grid: &ModelGrid) -> Result<(Vec<f32>, serde_json::Value, String), String> {
    let receipt_path = data.with_extension("json");
    let receipt_bytes = std::fs::read(&receipt_path)
        .map_err(|e| format!("cannot read the library window receipt {}: {e}", receipt_path.display()))?;
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_bytes)
        .map_err(|e| format!("{} is not JSON: {e}", receipt_path.display()))?;
    if receipt.get("schema").and_then(|v| v.as_str()) != Some(REF_SCHEMA)
        || receipt.get("status").and_then(|v| v.as_str()) != Some("READY")
    {
        return Err(format!("{} is not a READY {REF_SCHEMA} receipt", receipt_path.display()));
    }
    let stated = receipt
        .get("grid")
        .and_then(|g| g.get("identity_sha256"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let shape: Vec<usize> = receipt
        .get("shape")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    if stated != grid.identity_sha256 || shape != [grid.nz, grid.ny, grid.nx] {
        return Err(format!(
            "the library window {} was gridded onto {stated:?} {shape:?}, not this grid {:?} \
             [{}, {}, {}]: the column shapes would be taken from other columns",
            data.display(),
            grid.identity_sha256,
            grid.nz,
            grid.ny,
            grid.nx
        ));
    }
    let bytes = std::fs::read(data).map_err(|e| format!("cannot read {}: {e}", data.display()))?;
    if bytes.len() != grid.nz * grid.ny * grid.nx * 4 {
        return Err(format!("{} holds {} bytes, the grid needs {}", data.display(), bytes.len(), grid.nz * grid.ny * grid.nx * 4));
    }
    let digest = hex_sha256(&bytes);
    let declared = receipt.get("data").and_then(|d| d.get("sha256")).and_then(|v| v.as_str());
    if declared != Some(digest.as_str()) {
        return Err(format!("{} has sha256 {digest}, its receipt states {declared:?}", data.display()));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    Ok((values, receipt, digest))
}

// ---------------------------------------------------------------------------
// the request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompositeRequest {
    pub grid: PathBuf,
    pub frames: PathBuf,
    pub profile_from: Option<PathBuf>,
    pub profile_table: Option<PathBuf>,
    pub start: DateTime<Utc>,
    pub window_minutes: i64,
    pub windows: usize,
    pub out: PathBuf,
    pub echo_dbz: f32,
    pub clear_dbz: f32,
    pub oracle: bool,
    pub threads: usize,
}

/// `forecast`, or `oracle` when the frames are not causal, were issued after
/// the start, or are observations valid after the start.
pub fn lead_class(frames: &Frames, start: DateTime<Utc>, oracle: bool) -> &'static str {
    let future_observation =
        frames.kind == "observed" && frames.frames.iter().any(|(valid, _)| *valid > start);
    if oracle || !frames.causal || frames.issue_time > start || future_observation {
        "oracle"
    } else {
        "forecast"
    }
}

/// Hold the profile library's source window to the start.
///
/// The library sets every expanded column's vertical shape, so a library
/// built from a window that ends after the start carries observations the
/// forecast could not have had, while the windows it shapes say
/// `lead_class: forecast`.  A window that ends at or before the start is
/// causal.  With `--oracle` every window is labelled oracle anyway, so a
/// later window is allowed there.  `required` is true for `--profile-from`
/// (this stage reads the window, and `grid-ref` states its end whenever it
/// is given `--window-end`); a `--profile-table` built elsewhere may name no
/// window (a climatology), and is held to its end only when it states one.
pub fn check_profile_window_end(
    stated: Option<&serde_json::Value>,
    start: DateTime<Utc>,
    oracle: bool,
    what: &str,
    required: bool,
) -> Result<(), String> {
    let text = match stated {
        Some(serde_json::Value::String(text)) => Some(text.as_str()),
        None | Some(serde_json::Value::Null) => None,
        Some(other) => return Err(format!("{what} states its window end as {other}, not a time")),
    };
    let Some(text) = text else {
        if required && !oracle {
            return Err(format!(
                "{what} states no window end, so it cannot be shown to end at or before the \
                 start {}: its column shapes may come from observations the forecast could not \
                 have had. Grid it with --window-end, or give --oracle",
                iso8601(start)
            ));
        }
        return Ok(());
    };
    let end = parse_when(text).map_err(|e| format!("{what}: {e}"))?;
    if end > start && !oracle {
        return Err(format!(
            "{what} ends {}, after the start {}: the column shapes would come from \
             observations the forecast could not have had, while the windows say \
             lead_class forecast. Build the library from the window that ends at the start, \
             or give --oracle",
            iso8601(end),
            iso8601(start)
        ));
    }
    Ok(())
}

/// Run one grid-composite call; returns the stdout record.
pub fn grid_composite(request: &CompositeRequest) -> Result<String, Box<dyn Error>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(request.threads)
        .build()
        .map_err(|e| boxed_error(format!("cannot build a {}-thread pool: {e}", request.threads)))?;
    pool.install(|| run(request)).map_err(boxed_error)
}

fn run(request: &CompositeRequest) -> Result<String, String> {
    let all = Instant::now();
    if request.windows == 0 || request.window_minutes <= 0 {
        return Err("--windows and --window-minutes must be positive".to_string());
    }
    if !(request.echo_dbz > request.clear_dbz && request.echo_dbz > NOAA_ECHO_FLOOR_DBZ) {
        return Err(format!(
            "--echo-dbz {} must be above --clear-dbz {} and NOAA's 0.001 dBZ echo floor: a \
             column could otherwise be both echo and clear",
            request.echo_dbz, request.clear_dbz
        ));
    }
    let grid = ModelGrid::load(&request.grid)?;
    let frames = Frames::load(&request.frames)?;
    if !frames.causal && !request.oracle {
        return Err(format!(
            "the frames under {} are not causal (an input is stamped after the issue time, or \
             an observed frame is valid after it) and --oracle was not given: future \
             observations would be labelled as a forecast",
            request.frames.display()
        ));
    }
    let class = lead_class(&frames, request.start, request.oracle);
    if class == "oracle" && !request.oracle {
        return Err(format!(
            "the frames from {} ({}) are valid after the start {} or were issued after it, \
             and --oracle was not given: future observations would be labelled as a forecast",
            frames.id,
            frames.kind,
            iso8601(request.start)
        ));
    }
    let ends: Vec<DateTime<Utc>> = (1..=request.windows)
        .map(|n| request.start + Duration::minutes(request.window_minutes * n as i64))
        .collect();
    // Every window's frame is found before anything is written.
    let chosen: Vec<&(DateTime<Utc>, FrameEntry)> =
        ends.iter().map(|end| frames.frame_at(*end)).collect::<Result<_, _>>()?;

    let t0 = Instant::now();
    let mapping = map_columns(&grid, &frames.lattice);
    let mapped = mapping.iter().filter(|m| m.is_some()).count();
    let bins = level_height_bins(&grid);
    let map_seconds = t0.elapsed().as_secs_f64();

    std::fs::create_dir_all(&request.out)
        .map_err(|e| format!("cannot create {}: {e}", request.out.display()))?;
    let library = match (&request.profile_table, &request.profile_from) {
        (Some(table), _) => {
            let library = ProfileLibrary::load(table)?;
            check_profile_window_end(
                library.source.get("window_end"),
                request.start,
                request.oracle,
                &format!("--profile-table {}", table.display()),
                false,
            )?;
            library
        }
        (None, Some(window_path)) => {
            let (window, receipt, digest) = read_window(window_path, &grid)?;
            check_profile_window_end(
                receipt.get("window").and_then(|w| w.get("end")),
                request.start,
                request.oracle,
                &format!("the library window {}", window_path.display()),
                true,
            )?;
            let library = build_library(
                &window,
                grid.nz,
                grid.nx * grid.ny,
                &bins,
                serde_json::json!({
                    "window_data": window_path.display().to_string(),
                    "window_data_sha256": digest,
                    "window_end": receipt.get("window").and_then(|w| w.get("end")).cloned(),
                    "grid_identity_sha256": grid.identity_sha256,
                }),
            );
            if library.columns.at_or_above_30_dbz < MIN_LIBRARY_COLUMNS {
                return Err(format!(
                    "the library window {} holds {} columns at {LIBRARY_COUNT_DBZ} dBZ or more, \
                     fewer than {MIN_LIBRARY_COLUMNS}: the column shape would rest on a handful \
                     of storms. Give --profile-table to use a library built elsewhere",
                    window_path.display(),
                    library.columns.at_or_above_30_dbz
                ));
            }
            library
        }
        (None, None) => {
            return Err("give --profile-from (the observed 3D window at the start) or --profile-table".to_string())
        }
    };
    let library_text = format!(
        "{}\n",
        serde_json::to_string_pretty(&library).map_err(|e| e.to_string())?
    );
    let library_path = request.out.join("profile-library.json");
    write_atomic(&library_path, library_text.as_bytes())?;
    let library_sha256 = hex_sha256(library_text.as_bytes());

    let columns = grid.nx * grid.ny;
    let mut rows = Vec::with_capacity(ends.len());
    for (end, (valid, entry)) in ends.iter().zip(chosen) {
        let started = Instant::now();
        let members = frames.read(entry)?;
        let classes = reduce_columns(&members, frames.members, &mapping, request.echo_dbz, request.clear_dbz);
        drop(members);
        // Expand column by column into a column-major scratch, then
        // transpose into the (nz, ny, nx) product.
        let mut by_column = vec![NO_COVERAGE; columns * grid.nz];
        let weak: usize = by_column
            .par_chunks_mut(grid.nz)
            .zip(classes.par_iter())
            .zip(bins.par_iter())
            .map(|((out, class), levels)| match class {
                ColumnClass::Echo(c) => library.expand(*c, levels, out),
                ColumnClass::Clear => {
                    out.fill(OBSERVED_NO_ECHO);
                    0
                }
                ColumnClass::NoCoverage => 0,
            })
            .sum();
        let mut field = vec![NO_COVERAGE; columns * grid.nz];
        field.par_chunks_mut(columns).enumerate().for_each(|(k, level)| {
            for (c, slot) in level.iter_mut().enumerate() {
                *slot = by_column[c * grid.nz + k];
            }
        });
        drop(by_column);
        let mut census = [0u64; 3];
        for v in &field {
            if *v >= NOAA_ECHO_FLOOR_DBZ {
                census[0] += 1;
            } else if *v > -100.0 {
                census[1] += 1;
            } else {
                census[2] += 1;
            }
        }
        let mut column_census = [0u64; 3];
        let mut echo_without_profile = 0u64;
        for (c, class) in classes.iter().enumerate() {
            match class {
                ColumnClass::Echo(_) => {
                    column_census[0] += 1;
                    if field[c] <= -200.0 && (0..grid.nz).all(|k| field[k * columns + c] <= -200.0) {
                        echo_without_profile += 1;
                    }
                }
                ColumnClass::Clear => column_census[1] += 1,
                ColumnClass::NoCoverage => column_census[2] += 1,
            }
        }
        let mut bytes = Vec::with_capacity(field.len() * 4);
        for v in &field {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        drop(field);
        let dir = request.out.join(window_stamp(*end));
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let data_sha = hex_sha256(&bytes);
        write_atomic(&dir.join("ref.f32"), &bytes)?;
        let receipt = serde_json::json!({
            "schema": REF_SCHEMA,
            "status": "READY",
            "product": "2D reflectivity frames expanded onto model levels, ref2tten convention",
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
            "data": {"file": "ref.f32", "bytes": bytes.len(), "sha256": data_sha},
            "grid": {
                "name": grid.name,
                "identity_sha256": grid.identity_sha256,
                "descriptor": request.grid.display().to_string(),
                "map_proj": format!("{:?}", grid.projection.spec.kind).to_lowercase(),
                "nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
            },
            "reduce": "composite-profile",
            "window": {
                "end": iso8601(*end),
                "frame_valid": iso8601(*valid),
                "frame_stamp": entry.stamp,
                "rule": "the frame valid at the window end (observed frames: nearest within 120 s)",
            },
            "source": {
                "frames_root": request.frames.display().to_string(),
                "frames_receipt_sha256": frames.receipt_sha256,
                "frame_sha256": entry.sha256,
                "id": frames.id,
                "kind": frames.kind,
                "issue_time": iso8601(frames.issue_time),
                "causal": frames.causal,
                "lead_class": class,
                "lead_minutes": entry.lead_minutes,
                "members": frames.members,
                "member_reduction": "probability-matched mean over columns with finite values in at least half the members; one member is taken as it is",
                "thresholds": {
                    "echo_dbz": request.echo_dbz,
                    "clear_dbz": request.clear_dbz,
                    "clear_rule": "every member finite and below clear_dbz: the whole column is observed no echo (-99)",
                    "no_coverage_rule": "outside the lattice, NaN in more than half the members, or between the thresholds: -99999, the model decides",
                },
                "mapping": "nearest source cell of every model mass column, through each side's own projection and earth radius",
                "mapped_columns": mapped,
                "library": "profile-library.json",
                "library_sha256": library_sha256,
                "expansion": "levels where the library's echo fraction is at least 0.5 take composite + median offset, shifted so the column maximum equals the composite; other levels are no coverage",
            },
            "counts": {
                "echo_cells": census[0],
                "clear_cells": census[1],
                "no_coverage_cells": census[2],
                "echo_columns": column_census[0],
                "clear_columns": column_census[1],
                "no_coverage_columns": column_census[2],
                "echo_columns_without_profile": echo_without_profile,
                "weak_levels_to_model": weak,
            },
            "threads": rayon::current_num_threads(),
            "seconds": round3(started.elapsed().as_secs_f64()),
            "binary": format!("rw_nexrad {}", env!("CARGO_PKG_VERSION")),
        });
        let text = format!("{}\n", serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?);
        write_atomic(&dir.join("ref.json"), text.as_bytes())?;
        rows.push(serde_json::json!({
            "end": iso8601(*end),
            "directory": window_stamp(*end),
            "frame_valid": iso8601(*valid),
            "sha256": data_sha,
            "echo_columns": column_census[0],
            "clear_columns": column_census[1],
        }));
    }
    let index = serde_json::json!({
        "schema": WINDOWS_SCHEMA,
        "status": "READY",
        "producer": "rw_nexrad grid-composite",
        "grid": {"identity_sha256": grid.identity_sha256, "descriptor": request.grid.display().to_string()},
        "start": iso8601(request.start),
        "window_minutes": request.window_minutes,
        "lead_class": class,
        "source": {"id": frames.id, "kind": frames.kind, "frames_receipt_sha256": frames.receipt_sha256},
        "library_sha256": library_sha256,
        "windows": rows,
    });
    let index_text = format!("{}\n", serde_json::to_string_pretty(&index).map_err(|e| e.to_string())?);
    write_atomic(&request.out.join("windows.json"), index_text.as_bytes())?;
    let record = serde_json::json!({
        "schema": COMPOSITE_SCHEMA,
        "status": "READY",
        "out": request.out.display().to_string(),
        "lead_class": class,
        "windows": index["windows"],
        "library": {"path": library_path.display().to_string(), "sha256": library_sha256,
                    "columns": library.columns},
        "seconds": {"map": round3(map_seconds), "total": round3(all.elapsed().as_secs_f64())},
    });
    Ok(format!("{}\n", serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?))
}

pub const GRID_COMPOSITE_USAGE: &str = "\
usage: rw_nexrad grid-composite --grid GRID.json --frames FRAMES_ROOT
           (--profile-from OBS_WINDOW/ref.f32 | --profile-table LIB.json)
           --start TIME --window-minutes M --windows N --out WINDOWS_ROOT [OPTIONS]

  Turn 2D reflectivity frames (gpuwm-obs.nowcast-frames.v1: a nowcast, or the
  observed MRMS composite from `rw_mrms frames`) into radar latent heating
  windows on a model grid: WINDOWS_ROOT/<end>/ref.f32 and ref.json, the same
  gpuwm-obs.radar-tten-ref.v1 format `grid-ref` writes, one per window end
  START + n*M for n = 1..N, plus WINDOWS_ROOT/profile-library.json.

  --grid FILE             gpuwm-obs.radar-tten-grid.v1 descriptor
  --frames DIR            frames root holding nowcast.json
  --profile-from FILE     the observed 3D window at the start time (ref.f32
                          with ref.json beside it, on the same grid): the
                          same-day profile library is built from it. It must
                          state a window end at or before --start (else
                          --oracle)
  --profile-table FILE    a profile library built elsewhere instead
  --start TIME            run start; window n ends at START + n*M
  --window-minutes M      window length, whole minutes
  --windows N             window count
  --out DIR               windows root
  --echo-dbz DBZ          reduced value at or above this is echo (default 15)
  --clear-dbz DBZ         every member below this is observed clear (default 5)
  --oracle                accept frames that are not causal and label every
                          window lead_class oracle
  --threads N             worker threads (default: available cores, at most 64)
";

/// Parse the grid-composite arguments (everything after the subcommand).
pub fn parse_request(args: &[String]) -> Result<CompositeRequest, Box<dyn Error>> {
    let mut grid = None;
    let mut frames = None;
    let mut profile_from = None;
    let mut profile_table = None;
    let mut start = None;
    let mut window_minutes = None;
    let mut windows = None;
    let mut out = None;
    let mut echo_dbz = DEFAULT_ECHO_DBZ;
    let mut clear_dbz = DEFAULT_CLEAR_DBZ;
    let mut oracle = false;
    let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(64);
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let mut value = || -> Result<String, Box<dyn Error>> {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| boxed_error(format!("{flag} needs a value")))
        };
        let count = |raw: &str, name: &str| -> Result<i64, Box<dyn Error>> {
            match raw.parse::<i64>() {
                Ok(v) if v > 0 => Ok(v),
                _ => Err(boxed_error(format!("{name} must be a positive whole number, got {raw:?}"))),
            }
        };
        let dbz = |raw: &str, name: &str| -> Result<f32, Box<dyn Error>> {
            match raw.parse::<f32>() {
                Ok(v) if v.is_finite() => Ok(v),
                _ => Err(boxed_error(format!("{name} must be a finite dBZ value, got {raw:?}"))),
            }
        };
        match flag {
            "--grid" => grid = Some(PathBuf::from(value()?)),
            "--frames" => frames = Some(PathBuf::from(value()?)),
            "--profile-from" => profile_from = Some(PathBuf::from(value()?)),
            "--profile-table" => profile_table = Some(PathBuf::from(value()?)),
            "--start" => start = Some(parse_when(&value()?).map_err(boxed_error)?),
            "--window-minutes" => window_minutes = Some(count(&value()?, "--window-minutes")?),
            "--windows" => windows = Some(count(&value()?, "--windows")? as usize),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--echo-dbz" => echo_dbz = dbz(&value()?, "--echo-dbz")?,
            "--clear-dbz" => clear_dbz = dbz(&value()?, "--clear-dbz")?,
            "--oracle" => oracle = true,
            "--threads" => threads = count(&value()?, "--threads")? as usize,
            "--help" | "-h" => return Err(boxed_error(GRID_COMPOSITE_USAGE)),
            other => {
                return Err(boxed_error(format!(
                    "unknown grid-composite option {other:?}\n\n{GRID_COMPOSITE_USAGE}"
                )))
            }
        }
        index += 1;
    }
    Ok(CompositeRequest {
        grid: grid.ok_or_else(|| boxed_error("--grid is required"))?,
        frames: frames.ok_or_else(|| boxed_error("--frames is required"))?,
        profile_from,
        profile_table,
        start: start.ok_or_else(|| boxed_error("--start is required"))?,
        window_minutes: window_minutes.ok_or_else(|| boxed_error("--window-minutes is required"))?,
        windows: windows.ok_or_else(|| boxed_error("--windows is required"))?,
        out: out.ok_or_else(|| boxed_error("--out is required"))?,
        echo_dbz,
        clear_dbz,
        oracle,
        threads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambert_round_trips_and_matches_snyder_at_the_origin() {
        let l = LambertLattice::new(38.5, 38.5, -97.5, 38.5, -97.5, 6_371_229.0, (10, 10), (3000.0, 3000.0), (0.0, 0.0)).unwrap();
        let (x, y) = l.project(38.5, -97.5);
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6, "{x} {y}");
        for (lat, lon) in [(21.1, -122.7), (47.8, -60.9), (35.0, -97.5), (30.0, -80.0)] {
            let (x, y) = l.project(lat, lon);
            let (la, lo) = l.unproject(x, y);
            assert!((la - lat).abs() < 1e-9 && (lo - lon).abs() < 1e-9, "{lat},{lon} -> {la},{lo}");
        }
        // Two standard parallels give the secant cone.
        let s = LambertLattice::new(30.0, 60.0, -97.3, 35.3, -97.3, 6_370_000.0, (4, 4), (1.0, 1.0), (0.0, 0.0)).unwrap();
        assert!((s.n - 0.7155668).abs() < 1e-6, "{}", s.n);
    }

    #[test]
    fn pmm_of_identical_members_is_the_member_and_keeps_the_distribution() {
        let cells = 50;
        let one: Vec<f32> = (0..cells).map(|c| (c as f32 * 1.3) % 60.0).collect();
        let mapping: Vec<Option<u32>> = (0..cells as u32).map(Some).collect();
        let three: Vec<f32> = one.iter().chain(&one).chain(&one).copied().collect();
        let a = reduce_columns(&one, 1, &mapping, 15.0, 5.0);
        let b = reduce_columns(&three, 3, &mapping, 15.0, 5.0);
        assert_eq!(a, b);
    }
}
