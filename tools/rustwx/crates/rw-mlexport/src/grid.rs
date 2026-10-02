//! The run's own grid, described the way CF and xarray read it.
//!
//! A projected domain (Lambert conformal, Mercator, polar stereographic)
//! keeps `(y, x)` with 2-D `latitude`/`longitude` and, when the grid is
//! uniform in the CF projection's metres on WRF's 6,370 km sphere, 1-D
//! `y`/`x` coordinates and a `crs` grid-mapping variable, so cartopy and
//! MetPy place it without help.  A regular latitude-longitude grid (a
//! global tape, an unrotated MAP_PROJ 6 window) gets 1-D `latitude`
//! (ascending) and `longitude` dimensions, the WeatherBench 2 shape, in
//! 0..360 when it circles the globe.

use serde_json::{json, Map, Value};

use crate::frame::FileMeta;

/// WRF's sphere.
pub const EARTH_RADIUS_M: f64 = 6_370_000.0;
const RAD: f64 = std::f64::consts::PI / 180.0;

/// A regular latitude-longitude grid's 1-D axes and the order the source
/// rows and columns are read in to produce them.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Regular {
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    /// Source row for each output row.
    pub rows: Vec<usize>,
    /// Source column for each output column.
    pub cols: Vec<usize>,
    /// True when the grid circles the globe in longitude.
    pub periodic: bool,
    /// Source spacing (degrees).
    pub dlat: f64,
    pub dlon: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NativeGrid {
    pub nx: usize,
    pub ny: usize,
    pub regular: Option<Regular>,
    /// CF grid-mapping attributes, when the projection's metres fit.
    pub crs: Option<Map<String, Value>>,
    pub x: Option<Vec<f64>>,
    pub y: Option<Vec<f64>>,
    /// One phrase for the `horizontal_grid` attribute.
    pub description: String,
}

fn wrap180(d: f64) -> f64 {
    let d = (d + 180.0).rem_euclid(360.0) - 180.0;
    if d == -180.0 { 180.0 } else { d }
}

/// CF forward transform for one projection, metres on the sphere.
enum Forward {
    Lambert { n: f64, f: f64, rho0: f64, lon0: f64 },
    Mercator { k: f64, lon0: f64 },
    Polar { south: bool, k0: f64, lon0: f64 },
}

impl Forward {
    fn new(meta: &FileMeta) -> Option<(Forward, Map<String, Value>)> {
        let r = EARTH_RADIUS_M;
        match meta.map_proj {
            1 => {
                let (t1, t2) = (meta.truelat1 * RAD, meta.truelat2 * RAD);
                let n = if (meta.truelat1 - meta.truelat2).abs() > 1e-6 {
                    (t1.cos() / t2.cos()).ln()
                        / ((std::f64::consts::FRAC_PI_4 + t2 / 2.0).tan()
                            / (std::f64::consts::FRAC_PI_4 + t1 / 2.0).tan())
                        .ln()
                } else {
                    t1.sin()
                };
                let f = t1.cos() * (std::f64::consts::FRAC_PI_4 + t1 / 2.0).tan().powf(n) / n;
                let origin = if meta.moad_cen_lat != 0.0 { meta.moad_cen_lat } else { meta.cen_lat };
                let rho0 = r * f / (std::f64::consts::FRAC_PI_4 + origin * RAD / 2.0).tan().powf(n);
                let parallels = if (meta.truelat1 - meta.truelat2).abs() > 1e-6 {
                    json!([meta.truelat1, meta.truelat2])
                } else {
                    json!(meta.truelat1)
                };
                let mut attrs = Map::new();
                attrs.insert("grid_mapping_name".into(), json!("lambert_conformal_conic"));
                attrs.insert("standard_parallel".into(), parallels);
                attrs.insert("longitude_of_central_meridian".into(), json!(meta.stand_lon));
                attrs.insert("latitude_of_projection_origin".into(), json!(origin));
                Some((Forward::Lambert { n, f, rho0, lon0: meta.stand_lon }, attrs))
            }
            3 => {
                let k = (meta.truelat1 * RAD).cos();
                let mut attrs = Map::new();
                attrs.insert("grid_mapping_name".into(), json!("mercator"));
                attrs.insert("standard_parallel".into(), json!(meta.truelat1));
                attrs.insert("longitude_of_projection_origin".into(), json!(meta.stand_lon));
                Some((Forward::Mercator { k, lon0: meta.stand_lon }, attrs))
            }
            2 => {
                let south = meta.truelat1 < 0.0;
                let k0 = (1.0 + (meta.truelat1.abs() * RAD).sin()) / 2.0;
                let mut attrs = Map::new();
                attrs.insert("grid_mapping_name".into(), json!("polar_stereographic"));
                attrs.insert("straight_vertical_longitude_from_pole".into(), json!(meta.stand_lon));
                attrs.insert("latitude_of_projection_origin".into(), json!(if south { -90.0 } else { 90.0 }));
                attrs.insert("standard_parallel".into(), json!(meta.truelat1));
                Some((Forward::Polar { south, k0, lon0: meta.stand_lon }, attrs))
            }
            _ => None,
        }
    }

    fn apply(&self, lat: f64, lon: f64) -> (f64, f64) {
        let r = EARTH_RADIUS_M;
        match *self {
            Forward::Lambert { n, f, rho0, lon0 } => {
                let rho = r * f / (std::f64::consts::FRAC_PI_4 + lat * RAD / 2.0).tan().powf(n);
                let theta = n * wrap180(lon - lon0) * RAD;
                (rho * theta.sin(), rho0 - rho * theta.cos())
            }
            Forward::Mercator { k, lon0 } => (
                r * k * wrap180(lon - lon0) * RAD,
                r * k * (std::f64::consts::FRAC_PI_4 + lat * RAD / 2.0).tan().ln(),
            ),
            Forward::Polar { south, k0, lon0 } => {
                let d = wrap180(lon - lon0) * RAD;
                if south {
                    let rho = 2.0 * r * k0 * (std::f64::consts::FRAC_PI_4 + lat * RAD / 2.0).tan();
                    (rho * d.sin(), rho * d.cos())
                } else {
                    let rho = 2.0 * r * k0 * (std::f64::consts::FRAC_PI_4 - lat * RAD / 2.0).tan();
                    (rho * d.sin(), -rho * d.cos())
                }
            }
        }
    }
}

/// 1-D `x` and `y` (metres) when every point's projected position is within
/// 5% of a cell of the uniform lattice `x0 + i dx`, `y0 + j dy`.
fn fit_axes(meta: &FileMeta, forward: &Forward, xlat: &[f64], xlong: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let (nx, ny) = (meta.nx, meta.ny);
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    let mut projected = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let (x, y) = forward.apply(xlat[j * nx + i], xlong[j * nx + i]);
            sx += x - i as f64 * meta.dx;
            sy += y - j as f64 * meta.dy;
            projected.push((x, y));
        }
    }
    let count = (nx * ny) as f64;
    let (x0, y0) = (sx / count, sy / count);
    let tolerance = 0.05 * meta.dx.min(meta.dy);
    for j in 0..ny {
        for i in 0..nx {
            let (x, y) = projected[j * nx + i];
            if (x - (x0 + i as f64 * meta.dx)).abs() > tolerance
                || (y - (y0 + j as f64 * meta.dy)).abs() > tolerance
            {
                return None;
            }
        }
    }
    Some((
        (0..nx).map(|i| x0 + i as f64 * meta.dx).collect(),
        (0..ny).map(|j| y0 + j as f64 * meta.dy).collect(),
    ))
}

/// A MAP_PROJ 6 grid whose rows are latitudes and columns longitudes.
fn regular_axes(meta: &FileMeta, xlat: &[f64], xlong: &[f64]) -> Option<Regular> {
    if meta.map_proj != 6 || (meta.pole_lat - 90.0).abs() > 1e-6 || meta.pole_lon.abs() > 1e-6 {
        return None;
    }
    let (nx, ny) = (meta.nx, meta.ny);
    if nx < 2 || ny < 2 {
        return None;
    }
    for j in 0..ny {
        for i in 0..nx {
            if (xlat[j * nx + i] - xlat[j * nx]).abs() > 1e-4 || (wrap180(xlong[j * nx + i] - xlong[i])).abs() > 1e-4 {
                return None;
            }
        }
    }
    let lat_src: Vec<f64> = (0..ny).map(|j| xlat[j * nx]).collect();
    let lon_src: Vec<f64> = (0..nx).map(|i| xlong[i]).collect();
    let dlat = (lat_src[ny - 1] - lat_src[0]).abs() / (ny - 1) as f64;
    let dlon = wrap180(lon_src[1] - lon_src[0]).abs();
    let ascending = lat_src[ny - 1] > lat_src[0];
    let rows: Vec<usize> = if ascending { (0..ny).collect() } else { (0..ny).rev().collect() };
    let lat: Vec<f64> = rows.iter().map(|&j| lat_src[j]).collect();
    let periodic = ((nx as f64) * dlon - 360.0).abs() < 1e-3;
    // Longitude: 0..360 unless that breaks a box that crosses the prime
    // meridian, which keeps -180..180.
    let to360: Vec<f64> = lon_src.iter().map(|l| l.rem_euclid(360.0)).collect();
    let mut cols: Vec<usize> = (0..nx).collect();
    let lon: Vec<f64>;
    if periodic {
        cols.sort_by(|&a, &b| to360[a].total_cmp(&to360[b]));
        lon = cols.iter().map(|&i| to360[i]).collect();
    } else {
        let monotone360 = to360.windows(2).all(|w| w[1] > w[0]);
        if monotone360 {
            lon = to360;
        } else {
            lon = lon_src.iter().map(|&l| wrap180(l)).collect();
        }
    }
    Some(Regular { lat, lon, rows, cols, periodic, dlat, dlon })
}

/// Describe the native grid from the first frame's metadata and
/// coordinates.
pub fn describe(meta: &FileMeta, xlat: &[f64], xlong: &[f64]) -> NativeGrid {
    if let Some(regular) = regular_axes(meta, xlat, xlong) {
        let description = format!(
            "native regular latitude-longitude {:.4} x {:.4} deg",
            regular.dlat, regular.dlon
        );
        return NativeGrid { nx: meta.nx, ny: meta.ny, regular: Some(regular), crs: None, x: None, y: None, description };
    }
    if let Some((forward, mut attrs)) = Forward::new(meta) {
        if let Some((x, y)) = fit_axes(meta, &forward, xlat, xlong) {
            attrs.insert("earth_radius".into(), json!(EARTH_RADIUS_M));
            attrs.insert("false_easting".into(), json!(0.0));
            attrs.insert("false_northing".into(), json!(0.0));
            let name = attrs["grid_mapping_name"].as_str().unwrap_or("").to_string();
            return NativeGrid {
                nx: meta.nx,
                ny: meta.ny,
                regular: None,
                crs: Some(attrs),
                x: Some(x),
                y: Some(y),
                description: format!("native {name}, {:.1} m", meta.dx),
            };
        }
    }
    NativeGrid {
        nx: meta.nx,
        ny: meta.ny,
        regular: None,
        crs: None,
        x: None,
        y: None,
        description: format!("native MAP_PROJ {} with 2-D latitude and longitude", meta.map_proj),
    }
}

/// Reorder one `[ny, nx]` plane into a regular grid's ascending axes.
pub fn reorder<T: Copy>(plane: &[T], nx: usize, regular: &Regular) -> Vec<T> {
    let mut out = Vec::with_capacity(plane.len());
    for &j in &regular.rows {
        let row = &plane[j * nx..(j + 1) * nx];
        out.extend(regular.cols.iter().map(|&i| row[i]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(map_proj: i32, nx: usize, ny: usize, dx: f64) -> FileMeta {
        FileMeta {
            nx, ny, nz: 2, domain: "d01".into(), parent: None, map_proj,
            truelat1: 30.0, truelat2: 60.0, stand_lon: -98.0, cen_lat: 40.0, cen_lon: -98.0,
            moad_cen_lat: 40.0, pole_lat: 90.0, pole_lon: 0.0, dx, dy: dx,
            simulation_start: None, gpuwm_version: None, ic_source: None, ic_cycle: None,
            source_engine: None,
            history_preset: None, spec_bdy_width: None, bucket_mm: None,
        }
    }

    #[test]
    fn a_wrf_lambert_grid_fits_the_cf_projection() {
        // Build XLAT/XLONG with static-fields' transcription of WRF's own
        // Lambert, then check the CF forward transform puts them on a
        // uniform lattice of dx.
        use static_fields::projection::{GridSpec, ProjectedGrid, ProjectionKind};
        let (nx, ny, dx) = (40usize, 30usize, 12_000.0);
        let spec = GridSpec {
            kind: ProjectionKind::Lambert, ref_lat: 40.0, ref_lon: -98.0, truelat1: 30.0, truelat2: 60.0,
            stand_lon: -98.0, dx, dy: dx, e_we: nx as i64 + 1, e_sn: ny as i64 + 1,
            known_x: (nx as f64 + 1.0) / 2.0, known_y: (ny as f64 + 1.0) / 2.0,
            moad_cen_lat: 40.0, moad_cen_lon: -98.0, lat_deg: vec![], lon0_deg: 0.0, dlon_deg: 0.0,
        };
        let grid = ProjectedGrid::new(spec).unwrap();
        let mut xlat = Vec::new();
        let mut xlong = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let (lat, lon) = grid.ij_to_latlon(i as f64 + 1.0, j as f64 + 1.0);
                xlat.push(lat);
                xlong.push(lon);
            }
        }
        let native = describe(&meta(1, nx, ny, dx), &xlat, &xlong);
        assert!(native.crs.is_some(), "{}", native.description);
        let x = native.x.unwrap();
        assert!((x[1] - x[0] - dx).abs() < 1e-6);
    }

    #[test]
    fn a_global_tape_becomes_ascending_0_to_360() {
        let (nx, ny) = (8usize, 4usize);
        let mut xlat = Vec::new();
        let mut xlong = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                xlat.push(-67.5 + 45.0 * j as f64);
                xlong.push(-180.0 + 45.0 * i as f64);
            }
        }
        let mut m = meta(6, nx, ny, 5e6);
        m.truelat1 = 0.0;
        let native = describe(&m, &xlat, &xlong);
        let regular = native.regular.expect("regular");
        assert!(regular.periodic);
        assert_eq!(regular.lon, vec![0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]);
        assert_eq!(regular.cols[0], 4);
        let plane: Vec<i32> = (0..(nx * ny) as i32).collect();
        let out = reorder(&plane, nx, &regular);
        assert_eq!(&out[0..3], &[4, 5, 6]);
    }

    #[test]
    fn a_rotated_pole_is_not_regular_and_has_no_crs() {
        let (nx, ny) = (4usize, 3usize);
        let xlat: Vec<f64> = (0..nx * ny).map(|k| (k / nx) as f64).collect();
        let xlong: Vec<f64> = (0..nx * ny).map(|k| (k % nx) as f64).collect();
        let mut m = meta(6, nx, ny, 1e5);
        m.pole_lat = 40.0;
        let native = describe(&m, &xlat, &xlong);
        assert!(native.regular.is_none() && native.crs.is_none());
    }
}
