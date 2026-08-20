//! Render windows: the structured target grids an unstructured frame is
//! gathered onto, and the WRF projection attributes that describe them.
//!
//! Two kinds, because the renderer treats them differently. A Lambert window
//! emulates a WRF domain and gets the renderer's native regional map
//! furniture; a regular latitude/longitude window is the honest shape for a
//! global overview. The Lambert forward and inverse transforms below are WRF
//! `module_llxy.F` (`set_lc` and `ijll_lc`) transcribed, in f64, so the
//! emitted `XLAT`/`XLONG` agree with the projection attributes a reader
//! recomputes from.

use std::f64::consts::PI;

use crate::error::{MpasError, MpasResult};

/// WRF's spherical earth. These windows emulate a WRF domain, so they use
/// WRF's radius and not a WGS84 mean radius, or the emitted coordinates would
/// disagree with the projection attributes the renderer reads.
pub const WRF_EARTH_RADIUS_M: f64 = 6_370_000.0;
/// The radius the MPAS meshes carry, used for physical distances only.
pub const MPAS_EARTH_RADIUS_M: f64 = 6_371_229.0;

const RAD: f64 = PI / 180.0;
const DEG: f64 = 180.0 / PI;

/// A regular latitude/longitude window; renders under WRF `MAP_PROJ=6`.
#[derive(Debug, Clone, PartialEq)]
pub struct LatLonWindow {
    pub south: f64,
    pub north: f64,
    pub west: f64,
    pub east: f64,
    pub spacing_degrees: f64,
    pub description: String,
}

/// A Lambert conformal window emulating a WRF domain (`MAP_PROJ=1`).
#[derive(Debug, Clone, PartialEq)]
pub struct LambertWindow {
    pub centre_lat: f64,
    pub centre_lon: f64,
    pub dx_metres: f64,
    pub nx: usize,
    pub ny: usize,
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Window {
    LatLon(LatLonWindow),
    Lambert(LambertWindow),
}

/// A projection attribute value as it lands in the emitted file.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjAttr {
    Int(i32),
    Float(f64),
    Text(String),
}

impl Window {
    /// The catalogue of named windows. Two per mesh is the fidelity contract:
    /// a coarse global overview plus the focus region at its own resolution.
    pub fn named(name: &str) -> MpasResult<Window> {
        match name {
            "global" => Ok(Window::LatLon(LatLonWindow {
                south: -90.0,
                north: 90.0,
                west: -180.0,
                east: 179.75,
                spacing_degrees: 0.25,
                description: "global 0.25 degree overview window".to_string(),
            })),
            "focus" => Ok(Window::Lambert(LambertWindow {
                centre_lat: 37.0,
                centre_lon: -96.0,
                dx_metres: 22_000.0,
                nx: 240,
                ny: 150,
                truelat1: 30.0,
                truelat2: 60.0,
                stand_lon: -96.0,
                description:
                    "CONUS focus window on a 22 km Lambert grid, matched to the x4.163842 refined-region spacing"
                        .to_string(),
            })),
            other => Err(MpasError::Refusal(format!(
                "unknown render window '{other}' (known: focus, global)"
            ))),
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Window::LatLon(w) => &w.description,
            Window::Lambert(w) => &w.description,
        }
    }

    /// The canonical specification, as the weights digest and the emitted
    /// `MPAS_RENDER_WINDOW_SPEC` attribute serialize it: JSON object keys in
    /// sorted order, which is what makes the digest reproducible.
    pub fn spec_json(&self) -> String {
        match self {
            Window::LatLon(w) => format!(
                "{{\"east\": {}, \"kind\": \"latlon\", \"north\": {}, \"south\": {}, \"spacing_degrees\": {}, \"west\": {}}}",
                json_f64(w.east),
                json_f64(w.north),
                json_f64(w.south),
                json_f64(w.spacing_degrees),
                json_f64(w.west)
            ),
            Window::Lambert(w) => format!(
                "{{\"centre_lat\": {}, \"centre_lon\": {}, \"dx_metres\": {}, \"earth_radius_m\": {}, \"kind\": \"lambert\", \"nx\": {}, \"ny\": {}, \"stand_lon\": {}, \"truelat1\": {}, \"truelat2\": {}}}",
                json_f64(w.centre_lat),
                json_f64(w.centre_lon),
                json_f64(w.dx_metres),
                json_f64(WRF_EARTH_RADIUS_M),
                w.nx,
                w.ny,
                json_f64(w.stand_lon),
                json_f64(w.truelat1),
                json_f64(w.truelat2)
            ),
        }
    }

    /// The same specification with no spaces, which is the form the weights
    /// digest hashes (`json.dumps(separators=(",", ":"))`).
    pub fn spec_json_compact(&self) -> String {
        self.spec_json().replace(", ", ",").replace("\": ", "\":")
    }

    /// Target grid shape `(ny, nx)`.
    pub fn shape(&self) -> MpasResult<(usize, usize)> {
        match self {
            Window::Lambert(w) => Ok((w.ny, w.nx)),
            Window::LatLon(w) => {
                let lat = axis(w.south, w.north, w.spacing_degrees, "latitude")?;
                let lon = axis(w.west, w.east, w.spacing_degrees, "longitude")?;
                Ok((lat.len(), lon.len()))
            }
        }
    }

    /// Target latitudes and longitudes, row-major `(ny, nx)`, in degrees.
    pub fn coordinates(&self) -> MpasResult<(Vec<f64>, Vec<f64>)> {
        match self {
            Window::LatLon(w) => {
                let lat = axis(w.south, w.north, w.spacing_degrees, "latitude")?;
                let lon = axis(w.west, w.east, w.spacing_degrees, "longitude")?;
                if lat.iter().any(|v| v.abs() > 90.0) {
                    return Err(MpasError::Refusal(
                        "latitude window leaves [-90, 90]".to_string(),
                    ));
                }
                let mut lat_grid = Vec::with_capacity(lat.len() * lon.len());
                let mut lon_grid = Vec::with_capacity(lat.len() * lon.len());
                for &la in &lat {
                    for &lo in &lon {
                        lat_grid.push(la);
                        lon_grid.push(lo);
                    }
                }
                Ok((lat_grid, lon_grid))
            }
            Window::Lambert(w) => {
                if w.nx < 2 || w.ny < 2 {
                    return Err(MpasError::Refusal(
                        "a Lambert window needs at least 2x2 points".to_string(),
                    ));
                }
                if !w.dx_metres.is_finite() || w.dx_metres <= 0.0 {
                    return Err(MpasError::Refusal(
                        "Lambert dx must be positive and finite".to_string(),
                    ));
                }
                let projection = LambertProjection::set_lc(
                    w.truelat1,
                    w.truelat2,
                    w.stand_lon,
                    w.dx_metres,
                    w.centre_lat,
                    w.centre_lon,
                    0.5 * (w.nx as f64 + 1.0),
                    0.5 * (w.ny as f64 + 1.0),
                );
                let mut lat_grid = Vec::with_capacity(w.nx * w.ny);
                let mut lon_grid = Vec::with_capacity(w.nx * w.ny);
                for j in 1..=w.ny {
                    for i in 1..=w.nx {
                        let (la, lo) = projection.ijll(i as f64, j as f64);
                        lat_grid.push(la);
                        lon_grid.push(lo);
                    }
                }
                Ok((lat_grid, lon_grid))
            }
        }
    }

    /// The WRF global attributes describing this projection, in the order the
    /// emitted file carries them.
    pub fn projection_attributes(&self) -> MpasResult<Vec<(String, ProjAttr)>> {
        match self {
            Window::LatLon(w) => {
                let lat = axis(w.south, w.north, w.spacing_degrees, "latitude")?;
                let lon = axis(w.west, w.east, w.spacing_degrees, "longitude")?;
                let centre_lat = 0.5 * (lat[0] + lat[lat.len() - 1]);
                let centre_lon = 0.5 * (lon[0] + lon[lon.len() - 1]);
                // WRF writes DX/DY in metres even for a lat-lon grid; the
                // nominal spacing at the window centre is the honest number.
                let metres = w.spacing_degrees * RAD * WRF_EARTH_RADIUS_M;
                Ok(vec![
                    ("MAP_PROJ".into(), ProjAttr::Int(6)),
                    (
                        "MAP_PROJ_CHAR".into(),
                        ProjAttr::Text("Cylindrical Equidistant".into()),
                    ),
                    ("DX".into(), ProjAttr::Float(metres)),
                    ("DY".into(), ProjAttr::Float(metres)),
                    ("TRUELAT1".into(), ProjAttr::Float(0.0)),
                    ("TRUELAT2".into(), ProjAttr::Float(0.0)),
                    ("STAND_LON".into(), ProjAttr::Float(centre_lon)),
                    ("CEN_LAT".into(), ProjAttr::Float(centre_lat)),
                    ("CEN_LON".into(), ProjAttr::Float(centre_lon)),
                    ("MOAD_CEN_LAT".into(), ProjAttr::Float(centre_lat)),
                    ("POLE_LAT".into(), ProjAttr::Float(90.0)),
                    ("POLE_LON".into(), ProjAttr::Float(0.0)),
                ])
            }
            Window::Lambert(w) => {
                let (lat, lon) = self.coordinates()?;
                let centre_j = w.ny / 2;
                let centre_i = w.nx / 2;
                let at = centre_j * w.nx + centre_i;
                Ok(vec![
                    ("MAP_PROJ".into(), ProjAttr::Int(1)),
                    (
                        "MAP_PROJ_CHAR".into(),
                        ProjAttr::Text("Lambert Conformal".into()),
                    ),
                    ("DX".into(), ProjAttr::Float(w.dx_metres)),
                    ("DY".into(), ProjAttr::Float(w.dx_metres)),
                    ("TRUELAT1".into(), ProjAttr::Float(w.truelat1)),
                    ("TRUELAT2".into(), ProjAttr::Float(w.truelat2)),
                    ("STAND_LON".into(), ProjAttr::Float(w.stand_lon)),
                    ("CEN_LAT".into(), ProjAttr::Float(lat[at])),
                    ("CEN_LON".into(), ProjAttr::Float(lon[at])),
                    ("MOAD_CEN_LAT".into(), ProjAttr::Float(lat[at])),
                    ("POLE_LAT".into(), ProjAttr::Float(90.0)),
                    ("POLE_LON".into(), ProjAttr::Float(0.0)),
                ])
            }
        }
    }
}

/// Render a float the way `json.dumps` does, so the digest input matches.
fn json_f64(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e16 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

fn axis(first: f64, last: f64, step: f64, role: &str) -> MpasResult<Vec<f64>> {
    if !(first.is_finite() && last.is_finite() && step.is_finite()) {
        return Err(MpasError::Refusal(format!(
            "{role} window bounds must be finite"
        )));
    }
    if last <= first || step <= 0.0 {
        return Err(MpasError::Refusal(format!("{role} window must increase")));
    }
    let count = ((last - first) / step).round() as usize + 1;
    let mut values: Vec<f64> = (0..count).map(|k| first + k as f64 * step).collect();
    let end = values[count - 1];
    if (end - last).abs() > 1.0e-9 {
        return Err(MpasError::Refusal(format!(
            "{role} spacing does not exactly partition its bounds"
        )));
    }
    values[count - 1] = last;
    Ok(values)
}

/// WRF `module_llxy.F` Lambert state.
#[derive(Debug, Clone, Copy)]
pub struct LambertProjection {
    cone: f64,
    hemi: f64,
    rebydx: f64,
    polei: f64,
    polej: f64,
    truelat1: f64,
    truelat2: f64,
    stand_lon: f64,
}

fn lambert_cone(truelat1: f64, truelat2: f64) -> f64 {
    if (truelat1 - truelat2).abs() > 0.1 {
        let numerator = (truelat1 * RAD).cos().log10() - (truelat2 * RAD).cos().log10();
        let denominator = ((45.0 - truelat1.abs() / 2.0) * RAD).tan().log10()
            - ((45.0 - truelat2.abs() / 2.0) * RAD).tan().log10();
        numerator / denominator
    } else {
        (truelat1.abs() * RAD).sin()
    }
}

impl LambertProjection {
    /// WRF `set_lc`, transcribed.
    #[allow(clippy::too_many_arguments)]
    pub fn set_lc(
        truelat1: f64,
        truelat2: f64,
        stand_lon: f64,
        dx_metres: f64,
        known_lat: f64,
        known_lon: f64,
        known_i: f64,
        known_j: f64,
    ) -> Self {
        let cone = lambert_cone(truelat1, truelat2);
        let hemi = if truelat1 >= 0.0 { 1.0 } else { -1.0 };
        let rebydx = WRF_EARTH_RADIUS_M / dx_metres;
        let mut delta_lon = known_lon - stand_lon;
        if delta_lon > 180.0 {
            delta_lon -= 360.0;
        }
        if delta_lon < -180.0 {
            delta_lon += 360.0;
        }
        let ctl1r = (truelat1 * RAD).cos();
        let rsw = rebydx * ctl1r / cone
            * (((90.0 * hemi - known_lat) * RAD / 2.0).tan()
                / ((90.0 * hemi - truelat1) * RAD / 2.0).tan())
            .powf(cone);
        let arg = cone * (delta_lon * RAD);
        let polei = hemi * known_i - hemi * rsw * arg.sin();
        let polej = hemi * known_j + rsw * arg.cos();
        LambertProjection {
            cone,
            hemi,
            rebydx,
            polei,
            polej,
            truelat1,
            truelat2,
            stand_lon,
        }
    }

    /// WRF `ijll_lc`, transcribed. Returns `(latitude, longitude)` in degrees.
    pub fn ijll(&self, i: f64, j: f64) -> (f64, f64) {
        let hemi = self.hemi;
        let chi1 = (90.0 - hemi * self.truelat1) * RAD;
        let chi2 = (90.0 - hemi * self.truelat2) * RAD;
        let inew = hemi * i;
        let jnew = hemi * j;
        let xx = inew - self.polei;
        let yy = self.polej - jnew;
        let radius_squared = xx * xx + yy * yy;
        let radius = radius_squared.sqrt() / self.rebydx;
        let mut longitude = self.stand_lon + DEG * (hemi * xx).atan2(yy) / self.cone;
        longitude = (longitude + 360.0).rem_euclid(360.0);
        let chi = if (chi1 - chi2).abs() < 1.0e-12 {
            2.0 * ((radius / chi1.tan()).powf(1.0 / self.cone) * (chi1 * 0.5).tan()).atan()
        } else {
            2.0 * ((radius * self.cone / chi1.sin()).powf(1.0 / self.cone) * (chi1 * 0.5).tan())
                .atan()
        };
        let mut latitude = (90.0 - chi * DEG) * hemi;
        if radius_squared == 0.0 {
            latitude = hemi * 90.0;
            longitude = self.stand_lon;
        }
        if longitude > 180.0 {
            longitude -= 360.0;
        } else if longitude < -180.0 {
            longitude += 360.0;
        }
        (latitude, longitude)
    }
}
