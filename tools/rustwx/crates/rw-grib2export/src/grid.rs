//! GRIB2 Section 3 for a WRF-format history grid (WMO-No. 306 templates
//! 3.0, 3.10, 3.20 and 3.30).
//!
//! WRF projects on a sphere of radius 6,370,000 m (ARW technical note
//! [R1]).  The exporter encodes that sphere exactly (earth shape 1 with the
//! radius given), so a decoder places every point where WRF put it and no
//! spacing correction is needed (`grid_geometry=wrf-native-locations/v1`).
//! Rows are written south first, west to east: scanning mode 0x40 (+i, +j).
//! First and last point coordinates are the history's own XLAT/XLONG,
//! encoded in micro-degrees with longitude in [0, 360) (spec 4.1).

use wx_core::grib2::GridDefinition;

use crate::{refuse, Result};

/// WRF's sphere (ARW technical note [R1]).
pub const WRF_EARTH_RADIUS_M: f64 = 6_370_000.0;

/// Scanning mode: +i (west to east), +j (south to north), rows consecutive.
pub const SCAN_SOUTH_FIRST: u8 = 0x40;

/// The projection attributes of a history (WRF global attributes).
#[derive(Clone, Debug, Default)]
pub struct Projection {
    pub map_proj: i32,
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub pole_lat: f64,
    pub pole_lon: f64,
    pub dx: f64,
    pub dy: f64,
}

fn wrap360(lon: f64) -> f64 {
    let w = lon.rem_euclid(360.0);
    if w >= 360.0 { 0.0 } else { w }
}

/// Build Section 3 for an `nx` x `ny` grid with point coordinates `lat` and
/// `lon` (`[ny][nx]`, south row first).
pub fn definition(p: &Projection, nx: usize, ny: usize, lat: &[f64], lon: &[f64]) -> Result<GridDefinition> {
    let n = nx * ny;
    if lat.len() != n || lon.len() != n || nx < 2 || ny < 2 {
        return Err(refuse("XLAT/XLONG do not match the mass grid, so the GRIB grid cannot place any point"));
    }
    if !(p.dx > 0.0 && p.dy > 0.0) {
        return Err(refuse("DX/DY are absent or not positive, so the GRIB grid spacing would be wrong"));
    }
    let mut g = GridDefinition::default();
    g.nx = nx as u32;
    g.ny = ny as u32;
    g.lat1 = lat[0];
    g.lon1 = wrap360(lon[0]);
    g.lat2 = lat[n - 1];
    g.lon2 = wrap360(lon[n - 1]);
    g.scan_mode = SCAN_SOUTH_FIRST;
    let south = p.truelat1 < 0.0;
    match p.map_proj {
        1 => {
            g.template = 30;
            g.dx = p.dx;
            g.dy = p.dy;
            g.lad = p.truelat1;
            g.lov = wrap360(p.stand_lon);
            g.latin1 = p.truelat1;
            g.latin2 = if p.truelat2 == 0.0 { p.truelat1 } else { p.truelat2 };
            g.projection_center_flag = if south { 0x80 } else { 0 };
        }
        2 => {
            g.template = 20;
            g.dx = p.dx;
            g.dy = p.dy;
            g.lad = p.truelat1;
            g.lov = wrap360(p.stand_lon);
            g.projection_center_flag = if south { 0x80 } else { 0 };
        }
        3 => {
            g.template = 10;
            g.dx = p.dx;
            g.dy = p.dy;
            g.lad = p.truelat1;
        }
        6 => {
            if (p.pole_lat - 90.0).abs() > 1e-6 || p.pole_lon.abs() > 1e-6 && (p.pole_lon - 180.0).abs() > 1e-6 {
                return Err(refuse(
                    "a rotated latitude-longitude history (POLE_LAT != 90) needs grid template 3.1, which this writer does not encode; writing 3.0 would misplace every point",
                ));
            }
            g.template = 0;
            // Angular increments from the history's own coordinates.
            g.dx = (lon[nx - 1] - lon[0]).rem_euclid(360.0) / (nx - 1) as f64;
            g.dy = (lat[n - nx] - lat[0]).abs() / (ny - 1) as f64;
        }
        other => {
            return Err(refuse(format!(
                "MAP_PROJ {other} has no GRIB2 grid template in this exporter (Lambert 1, polar stereographic 2, Mercator 3, latitude-longitude 6 are encoded)"
            )));
        }
    }
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambert_south_first() {
        let p = Projection { map_proj: 1, truelat1: 38.5, truelat2: 38.5, stand_lon: -97.5, pole_lat: 90.0, pole_lon: 0.0, dx: 3000.0, dy: 3000.0 };
        let lat = vec![30.0, 30.0, 31.0, 31.0];
        let lon = vec![-100.0, -99.0, -100.0, -99.0];
        let g = definition(&p, 2, 2, &lat, &lon).unwrap();
        assert_eq!(g.template, 30);
        assert_eq!(g.scan_mode, 0x40);
        assert!((g.lon1 - 260.0).abs() < 1e-9);
        assert!((g.lov - 262.5).abs() < 1e-9);
        assert_eq!(g.projection_center_flag, 0);
    }

    #[test]
    fn rotated_latlon_is_refused() {
        let p = Projection { map_proj: 6, pole_lat: 40.0, dx: 1.0, dy: 1.0, ..Default::default() };
        let lat = vec![0.0; 4];
        let lon = vec![0.0; 4];
        assert!(definition(&p, 2, 2, &lat, &lon).is_err());
    }
}
