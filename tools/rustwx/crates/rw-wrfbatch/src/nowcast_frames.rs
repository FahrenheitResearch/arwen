//! Nowcast frames and heating windows, read for drawing on a run's grid.
//!
//! Two inputs the radar-heating evaluation has to look at beside the runs
//! it drives, neither of which is a history file:
//!
//! * a **frames root** in the `gpuwm-obs.nowcast-frames.v1` contract
//!   (`docs/nowcast-frames.md`): `nowcast.json` plus raw little-endian
//!   float32 frames of shape `[members, ny, nx]` on the source's own
//!   lattice.  StormScope writes one; `rw_mrms frames` writes the observed
//!   composite in the same form;
//! * a **heating window** in the `gpuwm-obs.radar-tten-ref.v1` format:
//!   `ref.json` plus `ref.f32`, shape `[nz, ny, nx]`, already on the
//!   model's mass grid in the NOAA ref2tten convention.
//!
//! A frame is mapped onto the run grid cell by cell through latitude and
//! longitude (each side in its own projection and earth radius) and the
//! members are reduced the way the window adapter reduces them: the
//! probability-matched mean over covered members, and no coverage where
//! more than half the members are uncovered.  A window is drawn as its
//! column maximum: the strongest echo in the column, observed clear where
//! the column holds clear and no echo, and no coverage otherwise.
//!
//! Nothing here regrids a history file or decides anything about heating;
//! it only reads two documented byte formats and puts them where a sheet
//! can draw them.

use std::path::{Path, PathBuf};

use chrono::{DateTime, NaiveDateTime, Utc};
use rustwx_ensemble::{MemberStack, NanPolicy, PmmTieRule, probability_matched_mean};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The frames receipt schema this reader accepts.
pub const FRAMES_SCHEMA: &str = "gpuwm-obs.nowcast-frames.v1";
/// The heating window receipt schema this reader accepts.
pub const WINDOW_SCHEMA: &str = "gpuwm-obs.radar-tten-ref.v1";
/// Window convention: the radar looked and saw no echo.
pub const OBSERVED_CLEAR_DBZ: f32 = -99.0;
/// Window convention: values at or below this are no coverage.
pub const NO_COVERAGE_LIMIT_DBZ: f32 = -200.0;
/// Window convention: the smallest value read as echo (radar_tten.py).
pub const ECHO_FLOOR_DBZ: f32 = 0.001;
/// A lattice's recorded corners must reproduce from its projection numbers
/// within this distance, or the lattice is refused.
pub const CORNER_TOLERANCE_M: f64 = 100.0;

/// `2026-10-01T18:30:00Z`, `2026-10-01T18:30Z` or `20261001T1830Z`.
pub fn parse_utc(text: &str) -> Result<DateTime<Utc>, String> {
    let text = text.trim();
    if let Ok(time) = DateTime::parse_from_rfc3339(text) {
        return Ok(time.with_timezone(&Utc));
    }
    for format in ["%Y-%m-%dT%H:%MZ", "%Y%m%dT%H%MZ", "%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%S"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(text, format) {
            return Ok(naive.and_utc());
        }
    }
    Err(format!("{text:?} is not a UTC time (2026-10-01T18:30Z or 20261001T1830Z)"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn number(value: &Value, key: &str, what: &str) -> Result<f64, String> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("{what} has no finite number {key:?}"))
}

fn count(value: &Value, key: &str, what: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .filter(|count| *count > 0)
        .map(|count| count as usize)
        .ok_or_else(|| format!("{what} has no positive whole number {key:?}"))
}

/// Great-circle distance in metres on a sphere of `radius_m`.
pub fn haversine_m(lat_a: f64, lon_a: f64, lat_b: f64, lon_b: f64, radius_m: f64) -> f64 {
    let (phi_a, phi_b) = (lat_a.to_radians(), lat_b.to_radians());
    let d_phi = phi_b - phi_a;
    let d_lambda = (lon_b - lon_a).to_radians();
    let h = (d_phi / 2.0).sin().powi(2) + phi_a.cos() * phi_b.cos() * (d_lambda / 2.0).sin().powi(2);
    2.0 * radius_m * h.sqrt().min(1.0).asin()
}

/// Longitude difference wrapped into (-180, 180].
fn wrap_degrees(delta: f64) -> f64 {
    let mut wrapped = (delta + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped == -180.0 {
        wrapped = 180.0;
    }
    wrapped
}

/// A Lambert conformal lattice on a sphere.
///
/// Projection coordinates are pyproj's `lcc` metres with the origin at
/// (`ref_lat`, `stand_lon`), the convention of a lattice described by
/// `+lat_0 +lon_0 +lat_1 +lat_2 +R`.  `x0_m`, `y0_m` are the coordinates of
/// cell (row 0, column 0); `dx_m` and `dy_m` are signed steps along columns
/// and rows.  `ref_lon` is recorded by the contract and checked only
/// through the corners.
#[derive(Clone, Debug, PartialEq)]
pub struct LambertLattice {
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub ref_lat: f64,
    pub ref_lon: f64,
    pub earth_radius_m: f64,
    pub nx: usize,
    pub ny: usize,
    pub dx_m: f64,
    pub dy_m: f64,
    pub x0_m: f64,
    pub y0_m: f64,
}

impl LambertLattice {
    fn cone(&self) -> (f64, f64, f64) {
        let phi1 = self.truelat1.to_radians();
        let phi2 = self.truelat2.to_radians();
        let t = |phi: f64| (std::f64::consts::FRAC_PI_4 + phi / 2.0).tan();
        let n = if (self.truelat1 - self.truelat2).abs() < 1e-9 {
            phi1.sin()
        } else {
            (phi1.cos() / phi2.cos()).ln() / (t(phi2) / t(phi1)).ln()
        };
        let f = phi1.cos() * t(phi1).powf(n) / n;
        let rho0 = self.earth_radius_m * f / t(self.ref_lat.to_radians()).powf(n);
        (n, f, rho0)
    }

    /// Projection metres of a point.
    pub fn forward(&self, lat: f64, lon: f64) -> (f64, f64) {
        let (n, f, rho0) = self.cone();
        let t = (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0).tan();
        let rho = self.earth_radius_m * f / t.powf(n);
        let theta = n * wrap_degrees(lon - self.stand_lon).to_radians();
        (rho * theta.sin(), rho0 - rho * theta.cos())
    }

    /// The point at projection metres `(x, y)`.
    pub fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let (n, f, rho0) = self.cone();
        let sign = n.signum();
        let dy = rho0 - y;
        let rho = sign * (x * x + dy * dy).sqrt();
        let theta = (sign * x).atan2(sign * dy);
        let lat = 2.0 * ((self.earth_radius_m * f / rho).powf(1.0 / n)).atan() - std::f64::consts::FRAC_PI_2;
        let lon = self.stand_lon + (theta / n).to_degrees();
        (lat.to_degrees(), wrap_degrees(lon))
    }
}

/// A regular latitude-longitude lattice.  `lat0`, `lon0` are the centre
/// of cell (row 0, column 0); `dlat` is negative for rows that run north to
/// south.
#[derive(Clone, Debug, PartialEq)]
pub struct LatLonLattice {
    pub lat0: f64,
    pub lon0: f64,
    pub dlat: f64,
    pub dlon: f64,
    pub nx: usize,
    pub ny: usize,
}

/// A frames source's lattice.
#[derive(Clone, Debug, PartialEq)]
pub enum Lattice {
    Lambert(LambertLattice),
    LatLon(LatLonLattice),
}

impl Lattice {
    /// The `lattice` block of a frames receipt, refused when its recorded
    /// corners do not reproduce from its own projection numbers.
    pub fn from_json(block: &Value) -> Result<Self, String> {
        let what = "the frames lattice";
        let kind = block.get("kind").and_then(Value::as_str).unwrap_or("");
        let lattice = match kind {
            "lambert" => {
                let lattice = LambertLattice {
                    truelat1: number(block, "truelat1", what)?,
                    truelat2: number(block, "truelat2", what)?,
                    stand_lon: number(block, "stand_lon", what)?,
                    ref_lat: number(block, "ref_lat", what)?,
                    ref_lon: number(block, "ref_lon", what)?,
                    earth_radius_m: number(block, "earth_radius_m", what)?,
                    nx: count(block, "nx", what)?,
                    ny: count(block, "ny", what)?,
                    dx_m: number(block, "dx_m", what)?,
                    dy_m: number(block, "dy_m", what)?,
                    x0_m: number(block, "x0_m", what)?,
                    y0_m: number(block, "y0_m", what)?,
                };
                if lattice.dx_m == 0.0 || lattice.dy_m == 0.0 || lattice.earth_radius_m <= 0.0 {
                    return Err(format!("{what}: dx_m, dy_m and earth_radius_m must be non-zero"));
                }
                Self::Lambert(lattice)
            }
            "latlon" => {
                let lattice = LatLonLattice {
                    lat0: number(block, "lat0", what)?,
                    lon0: number(block, "lon0", what)?,
                    dlat: number(block, "dlat", what)?,
                    dlon: number(block, "dlon", what)?,
                    nx: count(block, "nx", what)?,
                    ny: count(block, "ny", what)?,
                };
                if lattice.dlat == 0.0 || lattice.dlon == 0.0 {
                    return Err(format!("{what}: dlat and dlon must be non-zero"));
                }
                match block.get("row_order").and_then(Value::as_str) {
                    Some("north_to_south") if lattice.dlat > 0.0 => {
                        return Err(format!("{what}: row_order north_to_south needs a negative dlat"));
                    }
                    Some("south_to_north") if lattice.dlat < 0.0 => {
                        return Err(format!("{what}: row_order south_to_north needs a positive dlat"));
                    }
                    Some("north_to_south" | "south_to_north") | None => {}
                    Some(other) => return Err(format!("{what}: unknown row_order {other:?}")),
                }
                Self::LatLon(lattice)
            }
            other => return Err(format!("{what}: unknown kind {other:?} (lambert or latlon)")),
        };
        lattice.check_corners(block.get("corners_latlon"))?;
        Ok(lattice)
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Lambert(l) => (l.ny, l.nx),
            Self::LatLon(l) => (l.ny, l.nx),
        }
    }

    fn radius_m(&self) -> f64 {
        match self {
            Self::Lambert(l) => l.earth_radius_m,
            Self::LatLon(_) => 6_371_229.0,
        }
    }

    /// Latitude and longitude of the centre of cell `(row, column)`.
    pub fn cell_latlon(&self, row: usize, column: usize) -> (f64, f64) {
        match self {
            Self::Lambert(l) => l.inverse(l.x0_m + column as f64 * l.dx_m, l.y0_m + row as f64 * l.dy_m),
            Self::LatLon(l) => (l.lat0 + row as f64 * l.dlat, wrap_degrees(l.lon0 + column as f64 * l.dlon)),
        }
    }

    /// The lattice cell nearest a point, or `None` outside the lattice.
    pub fn locate(&self, lat: f64, lon: f64) -> Option<(usize, usize)> {
        if !(lat.is_finite() && lon.is_finite()) {
            return None;
        }
        let (row, column) = match self {
            Self::Lambert(l) => {
                let (x, y) = l.forward(lat, lon);
                (((y - l.y0_m) / l.dy_m).round(), ((x - l.x0_m) / l.dx_m).round())
            }
            Self::LatLon(l) => (
                ((lat - l.lat0) / l.dlat).round(),
                (wrap_degrees(lon - l.lon0) / l.dlon).round(),
            ),
        };
        let (ny, nx) = self.shape();
        if row < 0.0 || column < 0.0 || row >= ny as f64 || column >= nx as f64 {
            return None;
        }
        Some((row as usize, column as usize))
    }

    /// `corners_latlon` (docs/nowcast-frames.md): the centres of the four
    /// corner cells as `[lat, lon]`, keyed `j0_i0`, `j0_in` (row 0, last
    /// column), `jn_i0` (last row, column 0) and `jn_in`.
    fn check_corners(&self, corners: Option<&Value>) -> Result<(), String> {
        let corners = corners
            .and_then(Value::as_object)
            .ok_or("the frames lattice needs corners_latlon: {j0_i0, j0_in, jn_i0, jn_in} as [lat, lon]")?;
        let (ny, nx) = self.shape();
        for (name, row, column) in
            [("j0_i0", 0, 0), ("j0_in", 0, nx - 1), ("jn_i0", ny - 1, 0), ("jn_in", ny - 1, nx - 1)]
        {
            let pair = corners
                .get(name)
                .and_then(Value::as_array)
                .filter(|pair| pair.len() == 2)
                .and_then(|pair| Some((pair[0].as_f64()?, pair[1].as_f64()?)))
                .ok_or_else(|| format!("corners_latlon has no [lat, lon] for {name}"))?;
            let (lat, lon) = self.cell_latlon(row, column);
            let distance = haversine_m(lat, lon, pair.0, pair.1, self.radius_m());
            if !(distance <= CORNER_TOLERANCE_M) {
                return Err(format!(
                    "the frames lattice's corner {name} is recorded at ({:.5}, {:.5}) and its \
                     projection numbers put it at ({lat:.5}, {lon:.5}), {distance:.0} m apart \
                     (limit {CORNER_TOLERANCE_M} m): the frames would be drawn in the wrong place",
                    pair.0, pair.1
                ));
            }
        }
        Ok(())
    }
}

/// Spacing along a lattice axis must follow a straight line within this
/// distance, or the lattice is refused: the frames would be drawn, and
/// the heating applied, in the wrong columns.
pub const UNIFORM_TOLERANCE_M: f64 = 5.0;
/// The same rule for a latitude-longitude lattice, in degrees (about two
/// metres).
pub const UNIFORM_TOLERANCE_DEG: f64 = 2.0e-5;

/// The projection a Lambert lattice is fitted in: the numbers of the
/// contract's `lambert` block that cannot be read off a grid's own cell
/// centres.
#[derive(Clone, Debug, PartialEq)]
pub struct LambertProjection {
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub ref_lat: f64,
    pub earth_radius_m: f64,
}

impl LambertProjection {
    /// `truelat1=38.5,truelat2=38.5,stand_lon=-97.5,ref_lat=38.5,earth_radius_m=6371229`,
    /// every key once, in any order.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut fields: [Option<f64>; 5] = [None; 5];
        const KEYS: [&str; 5] = ["truelat1", "truelat2", "stand_lon", "ref_lat", "earth_radius_m"];
        for item in text.split(',').map(str::trim).filter(|item| !item.is_empty()) {
            let (key, value) = item
                .split_once('=')
                .ok_or_else(|| format!("projection item {item:?} is not key=value"))?;
            let index = KEYS
                .iter()
                .position(|known| *known == key.trim())
                .ok_or_else(|| format!("unknown projection key {key:?} (one of {})", KEYS.join(", ")))?;
            let number: f64 = value
                .trim()
                .parse()
                .ok()
                .filter(|number: &f64| number.is_finite())
                .ok_or_else(|| format!("projection {key} is not a number: {value:?}"))?;
            if fields[index].replace(number).is_some() {
                return Err(format!("projection key {key} given twice"));
            }
        }
        let missing: Vec<&str> =
            KEYS.iter().zip(&fields).filter(|(_, value)| value.is_none()).map(|(key, _)| *key).collect();
        if !missing.is_empty() {
            return Err(format!("projection is missing {}", missing.join(", ")));
        }
        let projection = Self {
            truelat1: fields[0].unwrap(),
            truelat2: fields[1].unwrap(),
            stand_lon: fields[2].unwrap(),
            ref_lat: fields[3].unwrap(),
            earth_radius_m: fields[4].unwrap(),
        };
        if projection.earth_radius_m <= 0.0 {
            return Err("projection earth_radius_m must be positive".into());
        }
        // Both hemispheres project (the cone constant carries the sign); what
        // cannot is a cone constant of zero, which every projected x and y
        // divides by: true latitudes on the equator, or a symmetric pair.
        let (n, _, _) = projection.template().cone();
        if !n.is_finite() || n.abs() < 1.0e-6 {
            return Err(format!(
                "true latitudes {} and {} give a degenerate cone (constant {n}); every projected \
                 coordinate would be infinite and no lattice could be fitted",
                projection.truelat1, projection.truelat2
            ));
        }
        Ok(projection)
    }

    /// A unit lattice at the projection origin, for projecting points.
    fn template(&self) -> LambertLattice {
        LambertLattice {
            truelat1: self.truelat1,
            truelat2: self.truelat2,
            stand_lon: self.stand_lon,
            ref_lat: self.ref_lat,
            ref_lon: self.stand_lon,
            earth_radius_m: self.earth_radius_m,
            nx: 1,
            ny: 1,
            dx_m: 1.0,
            dy_m: 1.0,
            x0_m: 0.0,
            y0_m: 0.0,
        }
    }
}

/// Least squares `v = v0 + k d` over `k = 0..n`, refused when any point
/// departs from that line by more than `tolerance`.
fn fit_uniform(values: &[f64], what: &str, tolerance: f64, unit: &str) -> Result<(f64, f64), String> {
    if values.len() < 2 {
        return Err(format!("{what} has fewer than two points, so the lattice spacing cannot be known"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{what} holds a coordinate that is not a number"));
    }
    let n = values.len() as f64;
    let mean_k = (n - 1.0) / 2.0;
    let mean_v = values.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (k, value) in values.iter().enumerate() {
        let dk = k as f64 - mean_k;
        sxy += dk * (value - mean_v);
        sxx += dk * dk;
    }
    let d = sxy / sxx;
    let v0 = mean_v - d * mean_k;
    let residual = values
        .iter()
        .enumerate()
        .map(|(k, value)| (value - (v0 + d * k as f64)).abs())
        .fold(0.0, f64::max);
    if !(residual <= tolerance) {
        return Err(format!(
            "{what} is not uniformly spaced (largest departure from a straight line {residual:.3} {unit}, \
             tolerance {tolerance} {unit}); the frames would be drawn in the wrong place"
        ));
    }
    if d == 0.0 {
        return Err(format!("{what} has zero spacing"));
    }
    Ok((v0, d))
}

fn round_to(value: f64, decimals: i32) -> f64 {
    let factor = 10f64.powi(decimals);
    (value * factor).round() / factor
}

fn check_plane_shape(lat: &[f64], lon: &[f64], ny: usize, nx: usize) -> Result<(), String> {
    if ny < 2 || nx < 2 {
        return Err(format!("a {ny}x{nx} grid is too small to carry a lattice"));
    }
    if lat.len() != ny * nx || lon.len() != ny * nx {
        return Err(format!(
            "latitude ({}) and longitude ({}) do not match the {ny}x{nx} grid",
            lat.len(),
            lon.len()
        ));
    }
    Ok(())
}

const CORNERS: [(&str, usize, usize); 4] = [("j0_i0", 0, 0), ("j0_in", 0, usize::MAX), ("jn_i0", usize::MAX, 0), ("jn_in", usize::MAX, usize::MAX)];

fn corner_cells(ny: usize, nx: usize) -> [(&'static str, usize, usize); 4] {
    CORNERS.map(|(name, j, i)| (name, if j == usize::MAX { ny - 1 } else { j }, if i == usize::MAX { nx - 1 } else { i }))
}

/// The given corner cells must be where the fitted lattice puts them.
fn check_given_corners(lattice: &Lattice, lat: &[f64], lon: &[f64]) -> Result<(), String> {
    let (ny, nx) = lattice.shape();
    for (name, j, i) in corner_cells(ny, nx) {
        let (fit_lat, fit_lon) = lattice.cell_latlon(j, i);
        let (given_lat, given_lon) = (lat[j * nx + i], lon[j * nx + i]);
        let miss = haversine_m(fit_lat, fit_lon, given_lat, given_lon, lattice.radius_m());
        if !(miss <= CORNER_TOLERANCE_M) {
            return Err(format!(
                "corner {name} reproduces {miss:.0} m from the grid's own coordinate ({given_lat:.5}, \
                 {given_lon:.5}); the frames would be drawn in the wrong place"
            ));
        }
    }
    Ok(())
}

/// The contract's `lambert` lattice from a grid's cell-centre latitude and
/// longitude (row-major `[ny, nx]`, longitude in any wrap) in a given
/// projection.  Only the four edges are projected.  The x of the first
/// and last rows must agree, as must the y of the first and last columns,
/// so the grid is rectilinear in the projection; then each axis is fitted
/// as a straight line and the four corners, recomputed from the fitted
/// numbers, must land within [`CORNER_TOLERANCE_M`] of the given ones.
pub fn lambert_lattice_from_latlon(
    lat: &[f64],
    lon: &[f64],
    ny: usize,
    nx: usize,
    projection: &LambertProjection,
) -> Result<LambertLattice, String> {
    check_plane_shape(lat, lon, ny, nx)?;
    let template = projection.template();
    let project = |j: usize, i: usize| template.forward(lat[j * nx + i], lon[j * nx + i]);
    let x_row = |j: usize| (0..nx).map(|i| project(j, i).0).collect::<Vec<_>>();
    let y_column = |i: usize| (0..ny).map(|j| project(j, i).1).collect::<Vec<_>>();
    let (x_first, x_last) = (x_row(0), x_row(ny - 1));
    let (y_first, y_last) = (y_column(0), y_column(nx - 1));
    for (a, b, what) in [
        (&x_first, &x_last, "x of the first and last rows"),
        (&y_first, &y_last, "y of the first and last columns"),
    ] {
        let gap = a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        if !(gap <= UNIFORM_TOLERANCE_M) {
            return Err(format!(
                "the {what} differ by up to {gap:.2} m, so the grid is not rectilinear in this projection; \
                 the frames would be drawn in the wrong place"
            ));
        }
    }
    let (x0, dx) = fit_uniform(&x_first, "the x coordinate", UNIFORM_TOLERANCE_M, "m")?;
    let (y0, dy) = fit_uniform(&y_first, "the y coordinate", UNIFORM_TOLERANCE_M, "m")?;
    let lattice = LambertLattice {
        truelat1: projection.truelat1,
        truelat2: projection.truelat2,
        stand_lon: projection.stand_lon,
        ref_lat: projection.ref_lat,
        ref_lon: projection.stand_lon,
        earth_radius_m: projection.earth_radius_m,
        nx,
        ny,
        dx_m: round_to(dx, 4),
        dy_m: round_to(dy, 4),
        x0_m: round_to(x0, 4),
        y0_m: round_to(y0, 4),
    };
    check_given_corners(&Lattice::Lambert(lattice.clone()), lat, lon)?;
    Ok(lattice)
}

/// The contract's `latlon` lattice from a grid's cell-centre latitude and
/// longitude (row-major `[ny, nx]`): latitude must be constant along each
/// row and longitude along each column, and each must be uniformly spaced.
pub fn latlon_lattice_from_latlon(lat: &[f64], lon: &[f64], ny: usize, nx: usize) -> Result<LatLonLattice, String> {
    check_plane_shape(lat, lon, ny, nx)?;
    for j in 0..ny {
        for i in 0..nx {
            let lat_gap = (lat[j * nx + i] - lat[j * nx]).abs();
            let lon_gap = wrap_degrees(lon[j * nx + i] - lon[i]).abs();
            if !(lat_gap <= UNIFORM_TOLERANCE_DEG && lon_gap <= UNIFORM_TOLERANCE_DEG) {
                return Err(format!(
                    "cell ({j}, {i}) is {lat_gap:.6} degrees of latitude off its row and {lon_gap:.6} \
                     degrees of longitude off its column, so the grid is not a latitude-longitude lattice"
                ));
            }
        }
    }
    let latitudes: Vec<f64> = (0..ny).map(|j| lat[j * nx]).collect();
    // Longitudes unwrapped against the first, so a row across the antimeridian is still a line.
    let longitudes: Vec<f64> = (0..nx).map(|i| lon[0] + wrap_degrees(lon[i] - lon[0])).collect();
    let (lat0, dlat) = fit_uniform(&latitudes, "the latitude", UNIFORM_TOLERANCE_DEG, "degrees")?;
    let (lon0, dlon) = fit_uniform(&longitudes, "the longitude", UNIFORM_TOLERANCE_DEG, "degrees")?;
    let lattice = LatLonLattice {
        lat0: round_to(lat0, 6),
        lon0: round_to(wrap_degrees(lon0), 6),
        dlat: round_to(dlat, 8),
        dlon: round_to(dlon, 8),
        nx,
        ny,
    };
    check_given_corners(&Lattice::LatLon(lattice.clone()), lat, lon)?;
    Ok(lattice)
}

/// The contract's `lattice` block for a lattice, corners included.
pub fn lattice_json(lattice: &Lattice) -> Value {
    let (ny, nx) = lattice.shape();
    let mut corners = serde_json::Map::new();
    for (name, j, i) in corner_cells(ny, nx) {
        let (lat, lon) = lattice.cell_latlon(j, i);
        corners.insert(name.to_string(), serde_json::json!([round_to(lat, 6), round_to(lon, 6)]));
    }
    match lattice {
        Lattice::Lambert(l) => serde_json::json!({
            "kind": "lambert", "truelat1": l.truelat1, "truelat2": l.truelat2, "stand_lon": l.stand_lon,
            "ref_lat": l.ref_lat, "ref_lon": l.ref_lon, "earth_radius_m": l.earth_radius_m,
            "nx": l.nx, "ny": l.ny, "dx_m": l.dx_m, "dy_m": l.dy_m, "x0_m": l.x0_m, "y0_m": l.y0_m,
            "row_order": if l.dy_m > 0.0 { "south_to_north" } else { "north_to_south" },
            "corners_latlon": corners,
        }),
        Lattice::LatLon(l) => serde_json::json!({
            "kind": "latlon", "lat0": l.lat0, "lon0": l.lon0, "dlat": l.dlat, "dlon": l.dlon,
            "nx": l.nx, "ny": l.ny,
            "row_order": if l.dlat > 0.0 { "south_to_north" } else { "north_to_south" },
            "corners_latlon": corners,
        }),
    }
}

/// The frames receipt schema for the input ledger beside it.
pub const INPUTS_SCHEMA: &str = "gpuwm-obs.nowcast-inputs.v1";

/// One frame to write: `values` is `[members, ny, nx]` in C order, dBZ,
/// NaN for no coverage.
pub struct FrameData<'a> {
    pub valid: DateTime<Utc>,
    pub values: &'a [f32],
}

/// What a frames root's receipt says about where its frames came from.
/// Every field lands in `nowcast.json` under the contract's key of the
/// same name; `converter` is the block that says how the frames were
/// made from their source file.
pub struct FramesProvenance {
    pub source: Value,
    pub sampler: Value,
    pub seed: Value,
    pub issue_time: DateTime<Utc>,
    pub latest_input_time: Option<DateTime<Utc>>,
    pub causal: bool,
    pub input_objects: Vec<Value>,
    pub history: Value,
    pub timing: Value,
    pub peak_device_bytes: Value,
    pub converter: Value,
}

/// The contract's directory name for a valid time.
pub fn frame_dirname(valid: DateTime<Utc>) -> String {
    valid.format("%Y%m%dT%H%MZ").to_string()
}

fn utc_text(time: DateTime<Utc>) -> String {
    time.format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let partial = path.with_extension(match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => format!("{ext}.partial"),
        None => "partial".to_string(),
    });
    std::fs::write(&partial, bytes).map_err(|error| format!("write {}: {error}", partial.display()))?;
    std::fs::rename(&partial, path)
        .map_err(|error| format!("rename {} into place: {error}", partial.display()))
}

/// Write a frames root in the `gpuwm-obs.nowcast-frames.v1` contract:
/// every frame under its valid-time directory (written `.partial` and
/// renamed into place), the input ledger, then the receipt last.  Returns
/// the receipt's path.
pub fn write_frames_root(
    root: &Path,
    lattice: &Lattice,
    members: usize,
    provenance: &FramesProvenance,
    frames: &[FrameData<'_>],
) -> Result<PathBuf, String> {
    let receipt_path = root.join("nowcast.json");
    if let Ok(bytes) = std::fs::read(&receipt_path) {
        let ready = serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|doc| doc.get("status").and_then(Value::as_str).map(|status| status == "READY"))
            .unwrap_or(false);
        if ready {
            return Err(format!(
                "{} is already READY; frames would change under a receipt something may already have read",
                receipt_path.display()
            ));
        }
    }
    if members == 0 {
        return Err("a frame set needs at least one member".into());
    }
    if frames.is_empty() {
        return Err("a frame set needs at least one frame".into());
    }
    let (ny, nx) = lattice.shape();
    let plane = members * ny * nx;
    let mut step: Option<i64> = None;
    for (index, frame) in frames.iter().enumerate() {
        if frame.values.len() != plane {
            return Err(format!(
                "the frame at {} holds {} values; [{members}, {ny}, {nx}] is {plane}",
                utc_text(frame.valid),
                frame.values.len()
            ));
        }
        if frame.valid < provenance.issue_time {
            return Err(format!(
                "the frame at {} is valid before the issue time {}; that is history, not a nowcast",
                utc_text(frame.valid),
                utc_text(provenance.issue_time)
            ));
        }
        if index > 0 {
            let gap = (frame.valid - frames[index - 1].valid).num_seconds();
            if gap <= 0 {
                return Err(format!("frames are not in increasing valid time at {}", utc_text(frame.valid)));
            }
            match step {
                None => step = Some(gap),
                Some(step) if step != gap => {
                    return Err(format!(
                        "the frames step {gap} s at {} after stepping {step} s; frames are listed with no gap",
                        utc_text(frame.valid)
                    ));
                }
                Some(_) => {}
            }
        }
    }
    let lattice_block = lattice_json(lattice);
    Lattice::from_json(&lattice_block).map_err(|error| format!("the lattice block does not read back: {error}"))?;
    std::fs::create_dir_all(root).map_err(|error| format!("create {}: {error}", root.display()))?;
    let mut rows = Vec::with_capacity(frames.len());
    for frame in frames {
        let dir = root.join(frame_dirname(frame.valid));
        std::fs::create_dir_all(&dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
        let mut bytes = Vec::with_capacity(plane * 4);
        for value in frame.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let file = dir.join("refc.f32");
        write_atomic(&file, &bytes)?;
        rows.push(serde_json::json!({
            "valid": utc_text(frame.valid),
            "lead_minutes": (frame.valid - provenance.issue_time).num_minutes(),
            "file": format!("{}/refc.f32", frame_dirname(frame.valid)),
            "shape": [members, ny, nx],
            "dtype": "float32-le",
            "sha256": sha256_hex(&bytes),
            "extra": {},
        }));
    }
    let inputs = serde_json::json!({
        "schema": INPUTS_SCHEMA,
        "objects": provenance.input_objects,
        "slots": [],
    });
    let inputs_bytes = serde_json::to_vec_pretty(&inputs).map_err(|error| error.to_string())?;
    write_atomic(&root.join("inputs.json"), &inputs_bytes)?;
    let receipt = serde_json::json!({
        "schema": FRAMES_SCHEMA,
        "status": "READY",
        "source": provenance.source,
        "sampler": provenance.sampler,
        "seed": provenance.seed,
        "issue_time": utc_text(provenance.issue_time),
        "latest_input_time": provenance.latest_input_time.map(utc_text),
        "causal": provenance.causal,
        "variable": "refc",
        "units": "dBZ",
        "members": members,
        "lattice": lattice_block,
        "frames": rows,
        "inputs": {"file": "inputs.json", "sha256": sha256_hex(&inputs_bytes)},
        "history": provenance.history,
        "timing": provenance.timing,
        "peak_device_bytes": provenance.peak_device_bytes,
        "converter": provenance.converter,
    });
    let receipt_bytes = serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?;
    write_atomic(&receipt_path, &receipt_bytes)?;
    Ok(receipt_path)
}

/// One frame row of a frames receipt.
#[derive(Clone, Debug)]
pub struct FrameRow {
    pub valid: DateTime<Utc>,
    pub lead_minutes: i64,
    pub file: PathBuf,
    pub sha256: String,
}

/// A frames root's receipt, checked.
#[derive(Clone, Debug)]
pub struct FramesReceipt {
    pub root: PathBuf,
    pub source_kind: String,
    pub source_id: String,
    pub issue_time: DateTime<Utc>,
    pub causal: bool,
    pub members: usize,
    pub lattice: Lattice,
    pub frames: Vec<FrameRow>,
    pub receipt_sha256: String,
}

/// Read and check `ROOT/nowcast.json`.
pub fn read_frames_receipt(root: &Path) -> Result<FramesReceipt, String> {
    let path = root.join("nowcast.json");
    let bytes = std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let doc: Value =
        serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    let what = path.display().to_string();
    match doc.get("schema").and_then(Value::as_str) {
        Some(FRAMES_SCHEMA) => {}
        other => return Err(format!("{what}: schema {other:?}, this reader takes {FRAMES_SCHEMA}")),
    }
    match doc.get("status").and_then(Value::as_str) {
        Some("READY") => {}
        other => return Err(format!("{what}: status {other:?}; only a READY receipt is drawn")),
    }
    let source = doc.get("source").cloned().unwrap_or(Value::Null);
    let issue_time = parse_utc(
        doc.get("issue_time").and_then(Value::as_str).ok_or(format!("{what}: no issue_time"))?,
    )?;
    let causal = doc
        .get("causal")
        .and_then(Value::as_bool)
        .ok_or(format!("{what}: no causal flag; a frame set must say whether it saw the future"))?;
    let members = count(&doc, "members", &what)?;
    let lattice = Lattice::from_json(doc.get("lattice").unwrap_or(&Value::Null))
        .map_err(|error| format!("{what}: {error}"))?;
    let (ny, nx) = lattice.shape();
    let rows = doc
        .get("frames")
        .and_then(Value::as_array)
        .filter(|rows| !rows.is_empty())
        .ok_or(format!("{what}: no frames"))?;
    let mut frames = Vec::with_capacity(rows.len());
    for row in rows {
        let valid = parse_utc(row.get("valid").and_then(Value::as_str).ok_or(format!("{what}: a frame has no valid time"))?)?;
        let shape: Vec<usize> = row
            .get("shape")
            .and_then(Value::as_array)
            .map(|dims| dims.iter().filter_map(Value::as_u64).map(|d| d as usize).collect())
            .unwrap_or_default();
        if shape != [members, ny, nx] {
            return Err(format!(
                "{what}: the frame at {valid} has shape {shape:?}, the receipt's members and \
                 lattice say [{members}, {ny}, {nx}]"
            ));
        }
        let dtype = row.get("dtype").and_then(Value::as_str).unwrap_or("float32-le");
        if !matches!(dtype, "float32-le" | "<f4" | "float32" | "f4") {
            return Err(format!("{what}: dtype {dtype:?}; frames are little-endian float32"));
        }
        let file = row.get("file").and_then(Value::as_str).ok_or(format!("{what}: a frame names no file"))?;
        let sha256 = row
            .get("sha256")
            .and_then(Value::as_str)
            .filter(|text| text.len() == 64)
            .ok_or(format!("{what}: the frame at {valid} carries no sha256"))?
            .to_ascii_lowercase();
        if frames.iter().any(|have: &FrameRow| have.valid == valid) {
            return Err(format!("{what}: two frames claim {valid}"));
        }
        frames.push(FrameRow {
            valid,
            lead_minutes: row.get("lead_minutes").and_then(Value::as_i64).unwrap_or(
                (valid - issue_time).num_minutes(),
            ),
            file: root.join(file),
            sha256,
        });
    }
    Ok(FramesReceipt {
        root: root.to_path_buf(),
        source_kind: source.get("kind").and_then(Value::as_str).unwrap_or("").to_string(),
        source_id: source.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
        issue_time,
        causal,
        members,
        lattice,
        frames,
        receipt_sha256: sha256_hex(&bytes),
    })
}

/// A frame on the run grid.
#[derive(Clone, Debug)]
pub struct FramePlane {
    pub values: Vec<f32>,
    pub valid: DateTime<Utc>,
    pub reduction: &'static str,
    pub covered_cells: usize,
    pub outside_cells: usize,
}

/// More than half the members uncovered: the reduced cell is uncovered.
fn majority_uncovered(missing: usize, members: usize) -> bool {
    2 * missing > members
}

/// The frame valid at `valid`, mapped onto the grid `(lat, lon)` and
/// member-reduced as the window adapter reduces it.
pub fn frame_on_grid(
    receipt: &FramesReceipt,
    valid: DateTime<Utc>,
    lat: &[f32],
    lon: &[f32],
    ny: usize,
    nx: usize,
) -> Result<FramePlane, String> {
    let row = receipt.frames.iter().find(|row| row.valid == valid).ok_or_else(|| {
        format!(
            "{} holds no frame valid at {} (frames: {})",
            receipt.root.display(),
            valid.to_rfc3339(),
            receipt.frames.iter().map(|row| row.valid.format("%H:%MZ").to_string()).collect::<Vec<_>>().join(", ")
        )
    })?;
    let bytes = std::fs::read(&row.file).map_err(|error| format!("read {}: {error}", row.file.display()))?;
    let (sny, snx) = receipt.lattice.shape();
    let plane = sny * snx;
    if bytes.len() != receipt.members * plane * 4 {
        return Err(format!(
            "{} holds {} bytes; [{}, {sny}, {snx}] float32 is {}",
            row.file.display(),
            bytes.len(),
            receipt.members,
            receipt.members * plane * 4
        ));
    }
    if sha256_hex(&bytes) != row.sha256 {
        return Err(format!("{}: the bytes do not hash to the receipt's sha256", row.file.display()));
    }
    if lat.len() != ny * nx || lon.len() != ny * nx {
        return Err("the run grid's coordinates do not match its shape".into());
    }
    let source = |member: usize, index: usize| -> f32 {
        let at = (member * plane + index) * 4;
        f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    };
    let located: Vec<Option<usize>> = lat
        .iter()
        .zip(lon)
        .map(|(lat, lon)| {
            receipt
                .lattice
                .locate(f64::from(*lat), f64::from(*lon))
                .map(|(row, column)| row * snx + column)
        })
        .collect();
    let outside_cells = located.iter().filter(|cell| cell.is_none()).count();
    let members: Vec<Vec<f64>> = (0..receipt.members)
        .map(|member| {
            located
                .iter()
                .map(|cell| cell.map_or(f64::NAN, |index| f64::from(source(member, index))))
                .collect()
        })
        .collect();
    let (values, reduction): (Vec<f32>, &'static str) = if receipt.members == 1 {
        (members[0].iter().map(|value| *value as f32).collect(), "single member")
    } else {
        // Cells uncovered in more than half the members are left out of
        // the reduction entirely (their pooled values too), as the window
        // adapter leaves them out: the mean is taken over covered cells only.
        let dropped: Vec<bool> = (0..ny * nx)
            .map(|cell| {
                let missing = members.iter().filter(|member| !member[cell].is_finite()).count();
                majority_uncovered(missing, receipt.members)
            })
            .collect();
        let members: Vec<Vec<f64>> = members
            .into_iter()
            .map(|plane| {
                plane
                    .into_iter()
                    .zip(&dropped)
                    .map(|(value, dropped)| if *dropped { f64::NAN } else { value })
                    .collect()
            })
            .collect();
        let stack = MemberStack::new(
            ny,
            nx,
            members.into_iter().enumerate().map(|(number, plane)| (number as u32, plane)).collect(),
        )
        .map_err(|error| format!("member stack: {error}"))?;
        let reduced = probability_matched_mean(&stack, NanPolicy::Mask, PmmTieRule::FlatIndex)
            .map_err(|error| format!("probability-matched mean: {error}"))?;
        (
            reduced
                .into_iter()
                .zip(&dropped)
                .map(|(value, dropped)| if *dropped { f32::NAN } else { value as f32 })
                .collect(),
            "probability-matched mean over covered members",
        )
    };
    let covered_cells = values.iter().filter(|value| value.is_finite()).count();
    Ok(FramePlane { values, valid, reduction, covered_cells, outside_cells })
}

/// A heating window drawn as its column maximum.
#[derive(Clone, Debug)]
pub struct WindowPlane {
    pub values: Vec<f32>,
    pub valid: DateTime<Utc>,
    pub ny: usize,
    pub nx: usize,
    pub nz: usize,
    pub lead_class: Option<String>,
    pub source_id: Option<String>,
    pub start: Option<DateTime<Utc>>,
    pub grid_identity_sha256: Option<String>,
    pub echo_columns: usize,
    pub clear_columns: usize,
    pub no_coverage_columns: usize,
    pub receipt: PathBuf,
}

/// The column value of one window column, in the window convention.
pub fn column_value(column: impl IntoIterator<Item = f32>) -> f32 {
    let mut echo = f32::NEG_INFINITY;
    let mut clear = false;
    for value in column {
        if value.is_nan() || value <= NO_COVERAGE_LIMIT_DBZ {
            continue;
        }
        if value >= ECHO_FLOOR_DBZ {
            echo = echo.max(value);
        } else if value > -100.0 {
            clear = true;
        }
    }
    if echo.is_finite() {
        echo
    } else if clear {
        OBSERVED_CLEAR_DBZ
    } else {
        f32::NAN
    }
}

/// Read a window (`DIR/ref.json` or the receipt itself) and reduce it to
/// its column maximum.
pub fn window_column_max(path: &Path) -> Result<WindowPlane, String> {
    let receipt_path = if path.is_dir() { path.join("ref.json") } else { path.to_path_buf() };
    let dir = receipt_path.parent().map(Path::to_path_buf).unwrap_or_default();
    let text = std::fs::read(&receipt_path).map_err(|error| format!("read {}: {error}", receipt_path.display()))?;
    let doc: Value =
        serde_json::from_slice(&text).map_err(|error| format!("{}: {error}", receipt_path.display()))?;
    let what = receipt_path.display().to_string();
    match doc.get("schema").and_then(Value::as_str) {
        Some(WINDOW_SCHEMA) => {}
        other => return Err(format!("{what}: schema {other:?}, this reader takes {WINDOW_SCHEMA}")),
    }
    if let Some(status) = doc.get("status").and_then(Value::as_str) {
        if status != "READY" {
            return Err(format!("{what}: status {status:?}; only a READY window is drawn"));
        }
    }
    let shape: Vec<usize> = doc
        .get("shape")
        .and_then(Value::as_array)
        .map(|dims| dims.iter().filter_map(Value::as_u64).map(|d| d as usize).collect())
        .unwrap_or_default();
    let [nz, ny, nx] = shape[..] else {
        return Err(format!("{what}: shape {shape:?} is not [nz, ny, nx]"));
    };
    let data = doc.get("data").cloned().unwrap_or(Value::Null);
    let named = data.get("file").and_then(Value::as_str).unwrap_or("ref.f32");
    let named = PathBuf::from(named);
    let data_path = if named.is_absolute() && named.is_file() {
        named
    } else {
        dir.join(named.file_name().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("ref.f32")))
    };
    let bytes = std::fs::read(&data_path).map_err(|error| format!("read {}: {error}", data_path.display()))?;
    if bytes.len() != nz * ny * nx * 4 {
        return Err(format!("{}: {} bytes, shape [{nz}, {ny}, {nx}] float32 is {}", data_path.display(), bytes.len(), nz * ny * nx * 4));
    }
    if let Some(want) = data.get("sha256").and_then(Value::as_str) {
        if sha256_hex(&bytes) != want.to_ascii_lowercase() {
            return Err(format!("{}: the bytes do not hash to the receipt's data.sha256", data_path.display()));
        }
    }
    let valid = doc
        .get("window")
        .and_then(|window| window.get("end"))
        .and_then(Value::as_str)
        .ok_or(format!("{what}: no window.end; a window with no time cannot share a sheet row"))
        .and_then(parse_utc)?;
    let plane = ny * nx;
    let value_at = |k: usize, cell: usize| -> f32 {
        let at = (k * plane + cell) * 4;
        f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    };
    let values: Vec<f32> = (0..plane).map(|cell| column_value((0..nz).map(|k| value_at(k, cell)))).collect();
    let echo_columns = values.iter().filter(|v| v.is_finite() && **v >= ECHO_FLOOR_DBZ).count();
    let clear_columns = values.iter().filter(|v| **v == OBSERVED_CLEAR_DBZ).count();
    let source = doc.get("source").cloned().unwrap_or(Value::Null);
    Ok(WindowPlane {
        no_coverage_columns: plane - echo_columns - clear_columns,
        values,
        valid,
        ny,
        nx,
        nz,
        lead_class: source.get("lead_class").and_then(Value::as_str).map(str::to_string),
        source_id: source.get("id").or_else(|| source.get("source_id")).and_then(Value::as_str).map(str::to_string),
        // grid-composite records the frames' issue time; a hand-made window may say start.
        start: source
            .get("issue_time")
            .or_else(|| source.get("start"))
            .and_then(Value::as_str)
            .and_then(|text| parse_utc(text).ok()),
        grid_identity_sha256: doc
            .get("grid")
            .and_then(|grid| grid.get("identity_sha256"))
            .and_then(Value::as_str)
            .map(str::to_string),
        echo_columns,
        clear_columns,
        receipt: receipt_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HRRR CONUS: first point 21.138123 N, 122.719528 W; last point
    /// 47.842195 N, 60.917193 W (the published grid definition).
    fn hrrr_like() -> LambertLattice {
        let mut lattice = LambertLattice {
            truelat1: 38.5,
            truelat2: 38.5,
            stand_lon: -97.5,
            ref_lat: 38.5,
            ref_lon: -97.5,
            earth_radius_m: 6_371_229.0,
            nx: 1799,
            ny: 1059,
            dx_m: 3000.0,
            dy_m: 3000.0,
            x0_m: 0.0,
            y0_m: 0.0,
        };
        let (x0, y0) = lattice.forward(21.138123, -122.719528);
        lattice.x0_m = x0;
        lattice.y0_m = y0;
        lattice
    }

    #[test]
    fn lambert_forward_and_inverse_round_trip() {
        let lattice = hrrr_like();
        for (lat, lon) in [(38.5, -97.5), (21.138, -122.72), (47.84, -60.92), (30.0, -90.0)] {
            let (x, y) = lattice.forward(lat, lon);
            let (back_lat, back_lon) = lattice.inverse(x, y);
            assert!((back_lat - lat).abs() < 1e-9 && (back_lon - lon).abs() < 1e-9, "{lat} {lon}");
        }
        let (x, y) = lattice.forward(38.5, -97.5);
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6);
    }

    #[test]
    fn the_hrrr_grid_reproduces_its_published_far_corner() {
        let lattice = Lattice::Lambert(hrrr_like());
        let (lat, lon) = lattice.cell_latlon(1058, 1798);
        let miss = haversine_m(lat, lon, 47.842195, -60.917193, 6_371_229.0);
        assert!(miss < 100.0, "far corner {lat} {lon} is {miss} m off");
        assert_eq!(lattice.locate(21.138123, -122.719528), Some((0, 0)));
        assert_eq!(lattice.locate(47.842195, -60.917193), Some((1058, 1798)));
        assert_eq!(lattice.locate(10.0, -122.7), None);
    }

    #[test]
    fn column_values_follow_the_window_convention() {
        assert_eq!(column_value([-99999.0, 12.0, 30.5, -99.0]), 30.5);
        assert_eq!(column_value([-99999.0, -99.0, -99999.0]), OBSERVED_CLEAR_DBZ);
        assert!(column_value([-99999.0, f32::NAN]).is_nan());
        // Weak returns below the echo floor are clear air to the heating.
        assert_eq!(column_value([-5.0, -99999.0]), OBSERVED_CLEAR_DBZ);
    }

    #[test]
    fn the_majority_rule_is_strict() {
        assert!(!majority_uncovered(4, 8));
        assert!(majority_uncovered(5, 8));
        assert!(majority_uncovered(1, 1));
    }

    fn hrrr_projection() -> LambertProjection {
        LambertProjection {
            truelat1: 38.5,
            truelat2: 38.5,
            stand_lon: -97.5,
            ref_lat: 38.5,
            earth_radius_m: 6_371_229.0,
        }
    }

    /// Cell centres of a lattice, as a runner would write them (float32).
    fn centres(lattice: &Lattice) -> (Vec<f64>, Vec<f64>) {
        let (ny, nx) = lattice.shape();
        let mut lat = Vec::with_capacity(ny * nx);
        let mut lon = Vec::with_capacity(ny * nx);
        for j in 0..ny {
            for i in 0..nx {
                let (la, lo) = lattice.cell_latlon(j, i);
                lat.push(f64::from(la as f32));
                lon.push(f64::from(lo as f32));
            }
        }
        (lat, lon)
    }

    #[test]
    fn a_lambert_lattice_is_fitted_back_from_its_own_float32_centres() {
        let mut truth = hrrr_like();
        truth.nx = 40;
        truth.ny = 30;
        truth.x0_m += 300.0 * 3000.0;
        truth.y0_m += 200.0 * 3000.0;
        let (lat, lon) = centres(&Lattice::Lambert(truth.clone()));
        let fitted = lambert_lattice_from_latlon(&lat, &lon, 30, 40, &hrrr_projection()).unwrap();
        assert!((fitted.dx_m - 3000.0).abs() < 0.05, "dx {}", fitted.dx_m);
        assert!((fitted.dy_m - 3000.0).abs() < 0.05, "dy {}", fitted.dy_m);
        assert!((fitted.x0_m - truth.x0_m).abs() < 1.0, "x0 {} vs {}", fitted.x0_m, truth.x0_m);
        assert!((fitted.y0_m - truth.y0_m).abs() < 1.0, "y0 {} vs {}", fitted.y0_m, truth.y0_m);
        let block = lattice_json(&Lattice::Lambert(fitted.clone()));
        assert_eq!(block["row_order"], "south_to_north");
        assert_eq!(Lattice::from_json(&block).unwrap(), Lattice::Lambert(fitted));

        // Rows that run north to south carry a negative dy and say so.
        let mut flipped = truth.clone();
        flipped.y0_m = truth.y0_m + 29.0 * 3000.0;
        flipped.dy_m = -3000.0;
        let (lat, lon) = centres(&Lattice::Lambert(flipped));
        let fitted = lambert_lattice_from_latlon(&lat, &lon, 30, 40, &hrrr_projection()).unwrap();
        assert!((fitted.dy_m + 3000.0).abs() < 0.05);
        assert_eq!(lattice_json(&Lattice::Lambert(fitted))["row_order"], "north_to_south");
    }

    #[test]
    fn a_southern_hemisphere_lattice_is_fitted_and_a_degenerate_cone_is_refused() {
        // A lattice south of the equator: the cone constant is negative and
        // the projection, fit and corner check all carry that sign.
        let projection = LambertProjection {
            truelat1: -30.0,
            truelat2: -30.0,
            stand_lon: 135.0,
            ref_lat: -30.0,
            earth_radius_m: 6_371_229.0,
        };
        let mut truth = projection.template();
        truth.nx = 40;
        truth.ny = 30;
        truth.dx_m = 3000.0;
        truth.dy_m = 3000.0;
        let (x0, y0) = truth.forward(-36.0, 128.0);
        truth.x0_m = x0;
        truth.y0_m = y0;
        let (lat, lon) = centres(&Lattice::Lambert(truth.clone()));
        assert!(lat.iter().all(|lat| *lat < 0.0));
        let fitted = lambert_lattice_from_latlon(&lat, &lon, 30, 40, &projection).unwrap();
        assert!((fitted.dx_m - 3000.0).abs() < 0.05 && (fitted.dy_m - 3000.0).abs() < 0.05, "{fitted:?}");
        assert!((fitted.x0_m - truth.x0_m).abs() < 1.0 && (fitted.y0_m - truth.y0_m).abs() < 1.0, "{fitted:?}");
        let block = lattice_json(&Lattice::Lambert(fitted.clone()));
        assert_eq!(block["row_order"], "south_to_north");
        assert_eq!(Lattice::from_json(&block).unwrap(), Lattice::Lambert(fitted));
        assert!(LambertProjection::parse("truelat1=-30,truelat2=-30,stand_lon=135,ref_lat=-30,earth_radius_m=6371229").is_ok());
        // True latitudes on the equator, or a symmetric pair, have no cone.
        for text in [
            "truelat1=0,truelat2=0,stand_lon=135,ref_lat=0,earth_radius_m=6371229",
            "truelat1=30,truelat2=-30,stand_lon=135,ref_lat=0,earth_radius_m=6371229",
        ] {
            assert!(LambertProjection::parse(text).unwrap_err().contains("degenerate cone"), "{text}");
        }
    }

    #[test]
    fn a_grid_that_is_not_rectilinear_in_the_projection_is_refused() {
        // A regular latitude-longitude grid is not a Lambert lattice.
        let regular = LatLonLattice { lat0: 36.0, lon0: -98.0, dlat: 0.05, dlon: 0.05, nx: 40, ny: 30 };
        let (lat, lon) = centres(&Lattice::LatLon(regular.clone()));
        let error = lambert_lattice_from_latlon(&lat, &lon, 30, 40, &hrrr_projection()).unwrap_err();
        assert!(error.contains("not rectilinear") || error.contains("not uniformly spaced"), "{error}");
        // But it is a latitude-longitude lattice, and reads back as itself.
        let fitted = latlon_lattice_from_latlon(&lat, &lon, 30, 40).unwrap();
        assert!((fitted.lat0 - 36.0).abs() < 1e-5 && (fitted.lon0 + 98.0).abs() < 1e-5);
        assert!((fitted.dlat - 0.05).abs() < 1e-6 && (fitted.dlon - 0.05).abs() < 1e-6);
        let block = lattice_json(&Lattice::LatLon(fitted.clone()));
        assert_eq!(Lattice::from_json(&block).unwrap(), Lattice::LatLon(fitted));
        // And a Lambert grid is not a latitude-longitude lattice.
        let (lat, lon) = centres(&Lattice::Lambert(hrrr_like()));
        let (lat, lon): (Vec<f64>, Vec<f64>) = (lat[..30 * 1799].to_vec(), lon[..30 * 1799].to_vec());
        assert!(latlon_lattice_from_latlon(&lat, &lon, 30, 1799).unwrap_err().contains("not a latitude-longitude lattice"));
    }

    #[test]
    fn the_projection_text_names_every_key_once() {
        let parsed =
            LambertProjection::parse("truelat1=38.5, truelat2=38.5,stand_lon=-97.5,ref_lat=38.5,earth_radius_m=6371229")
                .unwrap();
        assert_eq!(parsed, hrrr_projection());
        assert!(LambertProjection::parse("truelat1=38.5").unwrap_err().contains("missing"));
        assert!(LambertProjection::parse("truelat1=38.5,truelat1=38.5,truelat2=1,stand_lon=1,ref_lat=1,earth_radius_m=1")
            .unwrap_err()
            .contains("twice"));
        assert!(LambertProjection::parse("lat_1=38.5").unwrap_err().contains("unknown"));
    }

    #[test]
    fn a_written_root_reads_back_and_refuses_to_be_overwritten() {
        let root = std::env::temp_dir().join(format!("rw-nowcast-write-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let lattice = Lattice::LatLon(LatLonLattice { lat0: 36.0, lon0: -98.0, dlat: 0.05, dlon: 0.05, nx: 4, ny: 3 });
        let issue = parse_utc("2026-08-19T00:00Z").unwrap();
        let provenance = FramesProvenance {
            source: serde_json::json!({"kind": "nowcast", "id": "synthetic"}),
            sampler: Value::Null,
            seed: serde_json::json!(1),
            issue_time: issue,
            latest_input_time: Some(issue),
            causal: true,
            input_objects: vec![serde_json::json!({"stream": "mrms", "key": "x"})],
            history: Value::Null,
            timing: Value::Null,
            peak_device_bytes: Value::Null,
            converter: serde_json::json!({"tool": "test"}),
        };
        let a: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let b: Vec<f32> = (0..24).map(|v| v as f32 + 0.5).collect();
        let frames = [
            FrameData { valid: issue + chrono::Duration::minutes(30), values: &a },
            FrameData { valid: issue + chrono::Duration::minutes(60), values: &b },
        ];
        let receipt = write_frames_root(&root, &lattice, 2, &provenance, &frames).unwrap();
        let read = read_frames_receipt(&root).unwrap();
        assert_eq!(read.members, 2);
        assert_eq!(read.frames.len(), 2);
        assert_eq!(read.frames[1].lead_minutes, 60);
        assert_eq!(read.frames[0].file, root.join("20260819T0030Z").join("refc.f32"));
        assert!(read.causal);
        assert_eq!(read.lattice, lattice);
        let lat: Vec<f32> = (0..3).flat_map(|j| (0..4).map(move |_| 36.0 + 0.05 * j as f32)).collect();
        let lon: Vec<f32> = (0..3).flat_map(|_| (0..4).map(|i| -98.0 + 0.05 * i as f32)).collect();
        let plane = frame_on_grid(&read, issue + chrono::Duration::minutes(60), &lat, &lon, 3, 4).unwrap();
        assert_eq!(plane.covered_cells, 12);
        let doc: Value = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
        assert_eq!(doc["inputs"]["file"], "inputs.json");
        assert_eq!(doc["converter"]["tool"], "test");
        assert!(root.join("inputs.json").is_file());
        assert!(write_frames_root(&root, &lattice, 2, &provenance, &frames).unwrap_err().contains("already READY"));
        let _ = std::fs::remove_dir_all(&root);

        // A gap in the frames, a frame before the issue, and a short frame are refused.
        let gapped = [
            FrameData { valid: issue + chrono::Duration::minutes(30), values: &a },
            FrameData { valid: issue + chrono::Duration::minutes(90), values: &b },
            FrameData { valid: issue + chrono::Duration::minutes(120), values: &b },
        ];
        assert!(write_frames_root(&root, &lattice, 2, &provenance, &gapped).unwrap_err().contains("no gap"));
        let early = [FrameData { valid: issue - chrono::Duration::minutes(30), values: &a }];
        assert!(write_frames_root(&root, &lattice, 2, &provenance, &early).unwrap_err().contains("history"));
        let short = [FrameData { valid: issue, values: &a[..12] }];
        assert!(write_frames_root(&root, &lattice, 2, &provenance, &short).unwrap_err().contains("holds 12 values"));
        assert!(!root.join("nowcast.json").exists());
    }

    #[test]
    fn times_parse_in_every_contract_spelling() {
        let want = parse_utc("2026-10-01T18:30:00Z").unwrap();
        assert_eq!(parse_utc("2026-10-01T18:30Z").unwrap(), want);
        assert_eq!(parse_utc("20261001T1830Z").unwrap(), want);
        assert!(parse_utc("18:30").is_err());
    }
}
