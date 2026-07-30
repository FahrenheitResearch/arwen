use super::parser::GridDefinition;

/// Compute latitude and longitude arrays for every grid point.
/// Returns (lats, lons) with length nx*ny, stored in row-major order (j * nx + i).
pub fn grid_latlon(grid: &GridDefinition) -> (Vec<f64>, Vec<f64>) {
    let nx = grid.nx as usize;
    let ny = grid.ny as usize;
    let n = nx * ny;

    match grid.template {
        0 => latlon_grid(grid, nx, ny, n),
        1 => rotated_latlon_grid(grid, nx, ny, n),
        10 => mercator_grid(grid, nx, ny, n),
        20 => polar_stereo_grid(grid, nx, ny, n),
        30 => lambert_grid(grid, nx, ny, n),
        40 => gaussian_grid(grid, nx, ny, n),
        90 => space_view_grid(grid, nx, ny, n),
        _ => {
            // Return empty vectors for unknown templates
            (Vec::new(), Vec::new())
        }
    }
}

/// Template 3.0: Regular latitude/longitude grid.
fn latlon_grid(grid: &GridDefinition, nx: usize, ny: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lats = Vec::with_capacity(n);
    let mut lons = Vec::with_capacity(n);

    // Determine direction from scan_mode
    // Bit 2 (0x40): 0 = points in +i direction, 1 = -i
    // Bit 3 (0x80): 0 = points in -j direction, 1 = +j
    let dlat = if ny > 1 {
        (grid.lat2 - grid.lat1) / (ny as f64 - 1.0)
    } else {
        0.0
    };
    let dlon = if nx > 1 {
        (grid.lon2 - grid.lon1) / (nx as f64 - 1.0)
    } else {
        0.0
    };

    for j in 0..ny {
        let lat = grid.lat1 + j as f64 * dlat;
        for i in 0..nx {
            let lon = grid.lon1 + i as f64 * dlon;
            lats.push(lat);
            lons.push(lon);
        }
    }

    (lats, lons)
}

/// Template 3.10: Mercator projection.
/// Inverse projection from grid (i, j) to (lat, lon).
fn mercator_grid(grid: &GridDefinition, nx: usize, ny: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lats = Vec::with_capacity(n);
    let mut lons = Vec::with_capacity(n);

    let deg2rad = std::f64::consts::PI / 180.0;
    let rad2deg = 180.0 / std::f64::consts::PI;

    // Earth radius (WMO standard)
    let r = 6_371_229.0_f64;

    // LaD is the latitude where dx/dy match the specified grid spacing.
    // The Mercator scale factor at LaD: cos(LaD).
    let lad_rad = grid.lad * deg2rad;
    let cos_lad = lad_rad.cos();

    // The first grid point in Mercator projected coordinates
    let lat1_rad = grid.lat1 * deg2rad;
    let lon1_rad = grid.lon1 * deg2rad;

    let x0 = r * cos_lad * lon1_rad;
    let y0 = r * cos_lad * ((std::f64::consts::PI / 4.0 + lat1_rad / 2.0).tan()).ln();

    let dx = grid.dx;
    let dy = grid.dy;

    for j in 0..ny {
        for i in 0..nx {
            let x = x0 + i as f64 * dx;
            let y = y0 + j as f64 * dy;

            let lon = (x / (r * cos_lad)) * rad2deg;
            let lat = (2.0 * (y / (r * cos_lad)).exp().atan() - std::f64::consts::PI / 2.0)
                * rad2deg;

            lats.push(lat);
            lons.push(lon);
        }
    }

    (lats, lons)
}

/// Template 3.20: Polar Stereographic projection.
/// Inverse projection from grid (i, j) to (lat, lon).
fn polar_stereo_grid(grid: &GridDefinition, nx: usize, ny: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lats = Vec::with_capacity(n);
    let mut lons = Vec::with_capacity(n);

    let deg2rad = std::f64::consts::PI / 180.0;
    let rad2deg = 180.0 / std::f64::consts::PI;

    let r = 6_371_229.0_f64;

    let lov_rad = grid.lov * deg2rad;
    let lad_rad = grid.lad * deg2rad;

    // projection_center_flag bit 0: 0 = North Pole, 1 = South Pole
    let south_pole = (grid.projection_center_flag & 1) != 0;

    // Scale factor at the true latitude (LaD)
    // For a polar stereographic projection tangent at a pole and
    // with true latitude LaD, the scale factor is:
    //   k = (1 + sin(|LaD|)) / 2   [for standard polar stereo]
    let k = (1.0 + lad_rad.abs().sin()) / 2.0;

    // Project the first grid point to stereographic x,y so we know the origin offset
    let lat1_rad = grid.lat1 * deg2rad;
    let lon1_rad = grid.lon1 * deg2rad;

    let (x0, y0) = if south_pole {
        let t = (std::f64::consts::PI / 4.0 + lat1_rad / 2.0).tan();
        let rho = 2.0 * r * k * t;
        let theta = lon1_rad - lov_rad;
        (rho * theta.sin(), rho * theta.cos())
    } else {
        let t = (std::f64::consts::PI / 4.0 - lat1_rad / 2.0).tan();
        let rho = 2.0 * r * k * t;
        let theta = lon1_rad - lov_rad;
        (rho * theta.sin(), -rho * theta.cos())
    };

    let dx = grid.dx;
    let dy = grid.dy;

    for j in 0..ny {
        for i in 0..nx {
            let x = x0 + i as f64 * dx;
            let y = y0 + j as f64 * dy;

            let (lat, lon) = if south_pole {
                let rho = (x * x + y * y).sqrt();
                let lat = if rho.abs() < 1e-10 {
                    -90.0
                } else {
                    (2.0 * (rho / (2.0 * r * k)).atan() - std::f64::consts::PI / 2.0) * rad2deg
                };
                let lon = (lov_rad + x.atan2(y)) * rad2deg;
                (lat, lon)
            } else {
                let rho = (x * x + y * y).sqrt();
                let lat = if rho.abs() < 1e-10 {
                    90.0
                } else {
                    (std::f64::consts::PI / 2.0 - 2.0 * (rho / (2.0 * r * k)).atan()) * rad2deg
                };
                let lon = (lov_rad + x.atan2(-y)) * rad2deg;
                (lat, lon)
            };

            // Normalize longitude to [-180, 360)
            let lon = if lon > 360.0 {
                lon - 360.0
            } else if lon < -180.0 {
                lon + 360.0
            } else {
                lon
            };

            lats.push(lat);
            lons.push(lon);
        }
    }

    (lats, lons)
}

/// Template 3.40: Gaussian Latitude/Longitude grid.
/// Approximated as regular lat/lon spacing. For exact Gaussian latitudes,
/// one would need to compute roots of Legendre polynomials.
fn gaussian_grid(grid: &GridDefinition, nx: usize, ny: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    // Use the same logic as latlon_grid — the lat1/lat2/lon1/lon2 and
    // approximate dy have already been populated by the parser.
    latlon_grid(grid, nx, ny, n)
}

/// Template 3.30: Lambert Conformal Conic projection.
/// Inverse projection from grid (i, j) to (lat, lon).
fn lambert_grid(grid: &GridDefinition, nx: usize, ny: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lats = Vec::with_capacity(n);
    let mut lons = Vec::with_capacity(n);

    let deg2rad = std::f64::consts::PI / 180.0;
    let rad2deg = 180.0 / std::f64::consts::PI;

    // Earth radius (6371.229 km is WMO standard)
    let r = 6_371_229.0_f64;

    let lat1_rad = grid.latin1 * deg2rad;
    let lat2_rad = grid.latin2 * deg2rad;
    let lov_rad = grid.lov * deg2rad;

    // Compute n (cone constant)
    let n = if (grid.latin1 - grid.latin2).abs() < 1.0e-6 {
        lat1_rad.sin()
    } else {
        let num = (lat1_rad.cos()).ln() - (lat2_rad.cos()).ln();
        let den = ((std::f64::consts::PI / 4.0 + lat2_rad / 2.0).tan()).ln()
            - ((std::f64::consts::PI / 4.0 + lat1_rad / 2.0).tan()).ln();
        num / den
    };

    // F factor
    let f_val = (lat1_rad.cos() * (std::f64::consts::PI / 4.0 + lat1_rad / 2.0).tan().powf(n))
        / n;

    // rho0 - distance from pole for the first grid point's latitude
    let lat1_pt_rad = grid.lat1 * deg2rad;
    let rho0 = r * f_val / (std::f64::consts::PI / 4.0 + lat1_pt_rad / 2.0).tan().powf(n);

    let lon1_rad = grid.lon1 * deg2rad;
    let theta0 = n * (lon1_rad - lov_rad);

    // Grid origin in projected coordinates
    // The first grid point (0,0) maps to (lat1, lon1)
    // x0, y0 are the projected coordinates of the first grid point
    let x0 = rho0 * theta0.sin();
    let y0 = rho0 - rho0 * theta0.cos();

    // Note: dx, dy are in meters for Lambert grids
    let dx = grid.dx;
    let dy = grid.dy;

    for j in 0..ny {
        for i in 0..nx {
            let x = x0 + i as f64 * dx;
            // y increases upward in projection space
            let y = y0 + j as f64 * dy;

            // Inverse Lambert conformal
            let rho0_full = rho0;
            let xp = x;
            let yp = rho0_full - y;
            let rho = if n > 0.0 {
                (xp * xp + yp * yp).sqrt()
            } else {
                -(xp * xp + yp * yp).sqrt()
            };

            let theta = xp.atan2(yp);

            let lat = if rho.abs() < 1.0e-10 {
                if n > 0.0 { 90.0 } else { -90.0 }
            } else {
                (2.0 * ((r * f_val / rho.abs()).powf(1.0 / n)).atan()
                    - std::f64::consts::PI / 2.0)
                    * rad2deg
            };

            let lon = (lov_rad + theta / n) * rad2deg;

            // Normalize longitude to [-180, 360)
            let lon = if lon > 360.0 {
                lon - 360.0
            } else if lon < -180.0 {
                lon + 360.0
            } else {
                lon
            };

            lats.push(lat);
            lons.push(lon);
        }
    }

    (lats, lons)
}

/// Template 3.1: Rotated Latitude/Longitude grid.
///
/// First computes the regular lat/lon coordinates on the rotated grid,
/// then transforms them back to geographic (unrotated) coordinates using
/// the south pole of rotation.
fn rotated_latlon_grid(grid: &GridDefinition, nx: usize, ny: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    // Start with regular lat/lon on the rotated grid
    let (rot_lats, rot_lons) = latlon_grid(grid, nx, ny, n);

    let mut lats = Vec::with_capacity(n);
    let mut lons = Vec::with_capacity(n);

    let deg2rad = std::f64::consts::PI / 180.0;
    let rad2deg = 180.0 / std::f64::consts::PI;

    // South pole of the rotated grid
    let sp_lat = grid.south_pole_lat * deg2rad;
    let sp_lon = grid.south_pole_lon * deg2rad;
    let rot_angle = grid.rotation_angle * deg2rad;

    // North pole of rotated system is at (-sp_lat, sp_lon + pi)
    let alpha = -sp_lat; // latitude of rotated north pole
    let sin_alpha = alpha.sin();
    let cos_alpha = alpha.cos();

    for idx in 0..n {
        let rlat = rot_lats[idx] * deg2rad;
        let rlon = rot_lons[idx] * deg2rad - rot_angle;

        let sin_rlat = rlat.sin();
        let cos_rlat = rlat.cos();
        let sin_rlon = rlon.sin();
        let cos_rlon = rlon.cos();

        // Geographic latitude
        let lat_geo = (sin_rlat * sin_alpha + cos_rlat * cos_rlon * cos_alpha).asin();

        // Geographic longitude
        // Standard formula: lon = atan2(cos(rlat)*sin(rlon),
        //   cos(rlat)*cos(rlon)*sin(alpha) - sin(rlat)*cos(alpha)) + sp_lon
        let lon_geo = (cos_rlat * sin_rlon).atan2(
            cos_rlat * cos_rlon * sin_alpha - sin_rlat * cos_alpha
        ) + sp_lon;

        lats.push(lat_geo * rad2deg);

        // Normalize longitude to [-180, 360)
        let mut lon_deg = lon_geo * rad2deg;
        while lon_deg < -180.0 {
            lon_deg += 360.0;
        }
        while lon_deg >= 360.0 {
            lon_deg -= 360.0;
        }
        lons.push(lon_deg);
    }

    (lats, lons)
}

/// Transform a single point from rotated to geographic coordinates.
///
/// `south_pole_lat` and `south_pole_lon` are in degrees.
/// `rotation_angle` is in degrees.
/// Returns (geo_lat, geo_lon) in degrees.
pub fn rotated_to_geographic(
    rot_lat: f64,
    rot_lon: f64,
    south_pole_lat: f64,
    south_pole_lon: f64,
    rotation_angle: f64,
) -> (f64, f64) {
    let deg2rad = std::f64::consts::PI / 180.0;
    let rad2deg = 180.0 / std::f64::consts::PI;

    let sp_lat = south_pole_lat * deg2rad;
    let sp_lon = south_pole_lon * deg2rad;

    let rlat = rot_lat * deg2rad;
    let rlon = (rot_lon - rotation_angle) * deg2rad;

    let alpha = -sp_lat;
    let sin_alpha = alpha.sin();
    let cos_alpha = alpha.cos();

    let sin_rlat = rlat.sin();
    let cos_rlat = rlat.cos();
    let sin_rlon = rlon.sin();
    let cos_rlon = rlon.cos();

    let lat_geo = (sin_rlat * sin_alpha + cos_rlat * cos_rlon * cos_alpha).asin();
    let lon_geo = (cos_rlat * sin_rlon).atan2(
        cos_rlat * cos_rlon * sin_alpha - sin_rlat * cos_alpha
    ) + sp_lon;

    let mut lon_deg = lon_geo * rad2deg;
    while lon_deg < -180.0 {
        lon_deg += 360.0;
    }
    while lon_deg >= 360.0 {
        lon_deg -= 360.0;
    }

    (lat_geo * rad2deg, lon_deg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser::GridDefinition;

    // ---- Template 0: Regular lat/lon grid ----

    #[test]
    fn test_latlon_grid_1x1() {
        let grid = GridDefinition {
            template: 0,
            nx: 1,
            ny: 1,
            lat1: 45.0,
            lon1: -90.0,
            lat2: 45.0,
            lon2: -90.0,
            ..Default::default()
        };
        let (lats, lons) = grid_latlon(&grid);
        assert_eq!(lats.len(), 1);
        assert_eq!(lons.len(), 1);
        assert!((lats[0] - 45.0).abs() < 1e-10);
        assert!((lons[0] - (-90.0)).abs() < 1e-10);
    }

    #[test]
    fn test_latlon_grid_2x2() {
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 10.0,
            lon2: 10.0,
            ..Default::default()
        };
        let (lats, lons) = grid_latlon(&grid);
        assert_eq!(lats.len(), 4);
        // Row 0: lat=0, Row 1: lat=10
        assert!((lats[0] - 0.0).abs() < 1e-10);
        assert!((lons[0] - 0.0).abs() < 1e-10);
        assert!((lats[1] - 0.0).abs() < 1e-10);
        assert!((lons[1] - 10.0).abs() < 1e-10);
        assert!((lats[2] - 10.0).abs() < 1e-10);
        assert!((lons[2] - 0.0).abs() < 1e-10);
        assert!((lats[3] - 10.0).abs() < 1e-10);
        assert!((lons[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_latlon_grid_3x3_global() {
        let grid = GridDefinition {
            template: 0,
            nx: 3,
            ny: 3,
            lat1: -90.0,
            lon1: 0.0,
            lat2: 90.0,
            lon2: 360.0,
            ..Default::default()
        };
        let (lats, lons) = grid_latlon(&grid);
        assert_eq!(lats.len(), 9);
        assert!((lats[0] - (-90.0)).abs() < 1e-10);
        assert!((lats[3] - 0.0).abs() < 1e-10);
        assert!((lats[6] - 90.0).abs() < 1e-10);
        assert!((lons[0] - 0.0).abs() < 1e-10);
        assert!((lons[1] - 180.0).abs() < 1e-10);
        assert!((lons[2] - 360.0).abs() < 1e-10);
    }

    // ---- Template 30: Lambert Conformal ----

    #[test]
    fn test_lambert_grid_first_point() {
        let grid = GridDefinition {
            template: 30,
            nx: 3,
            ny: 3,
            lat1: 21.138,
            lon1: 237.28,
            dx: 3000.0,
            dy: 3000.0,
            latin1: 38.5,
            latin2: 38.5,
            lov: 262.5,
            ..Default::default()
        };
        let (lats, lons) = grid_latlon(&grid);
        assert_eq!(lats.len(), 9);
        assert!((lats[0] - 21.138).abs() < 0.01, "lat[0]={}", lats[0]);
        assert!((lons[0] - 237.28).abs() < 0.01, "lon[0]={}", lons[0]);
    }

    // ---- Unknown template ----

    #[test]
    fn test_unknown_template_returns_empty() {
        let grid = GridDefinition {
            template: 999,
            nx: 5,
            ny: 5,
            ..Default::default()
        };
        let (lats, lons) = grid_latlon(&grid);
        assert!(lats.is_empty());
        assert!(lons.is_empty());
    }

    // ---- rotated_to_geographic ----

    #[test]
    fn test_rotated_to_geographic_identity() {
        // South pole at actual south pole => identity transform
        let (lat, lon) = rotated_to_geographic(45.0, 90.0, -90.0, 0.0, 0.0);
        assert!((lat - 45.0).abs() < 1e-6, "lat={}", lat);
        assert!((lon - 90.0).abs() < 1e-6, "lon={}", lon);
    }

    #[test]
    fn test_rotated_to_geographic_pole() {
        let (lat, _lon) = rotated_to_geographic(90.0, 0.0, -90.0, 0.0, 0.0);
        assert!((lat - 90.0).abs() < 1e-6, "lat={}", lat);
    }

    #[test]
    fn test_rotated_to_geographic_equator() {
        let (lat, lon) = rotated_to_geographic(0.0, 0.0, -90.0, 0.0, 0.0);
        assert!((lat - 0.0).abs() < 1e-6, "lat={}", lat);
        assert!((lon - 0.0).abs() < 1e-6, "lon={}", lon);
    }

    // ---- Gaussian grid (template 40) ----

    #[test]
    fn test_gaussian_grid_same_as_latlon() {
        let grid = GridDefinition {
            template: 40,
            nx: 2,
            ny: 2,
            lat1: -10.0,
            lon1: 0.0,
            lat2: 10.0,
            lon2: 20.0,
            ..Default::default()
        };
        let (lats, lons) = grid_latlon(&grid);
        assert_eq!(lats.len(), 4);
        assert!((lats[0] - (-10.0)).abs() < 1e-10);
        assert!((lats[2] - 10.0).abs() < 1e-10);
        assert!((lons[0] - 0.0).abs() < 1e-10);
        assert!((lons[1] - 20.0).abs() < 1e-10);
    }

    // ---- Polar stereographic (template 20) ----

    #[test]
    fn test_polar_stereo_grid_size() {
        let grid = GridDefinition {
            template: 20,
            nx: 3,
            ny: 3,
            lat1: 60.0,
            lon1: -120.0,
            dx: 10000.0,
            dy: 10000.0,
            lov: -100.0,
            lad: 60.0,
            projection_center_flag: 0, // north pole
            ..Default::default()
        };
        let (lats, lons) = grid_latlon(&grid);
        assert_eq!(lats.len(), 9);
        assert_eq!(lons.len(), 9);
        // All lats should be in a reasonable range for a north polar stereo
        for &lat in &lats {
            assert!(lat > 0.0 && lat <= 90.0, "unexpected lat={}", lat);
        }
    }

    // ---- Mercator (template 10) ----

    #[test]
    fn test_mercator_grid_first_point() {
        let grid = GridDefinition {
            template: 10,
            nx: 2,
            ny: 2,
            lat1: 20.0,
            lon1: -100.0,
            dx: 10000.0,
            dy: 10000.0,
            lad: 20.0,
            ..Default::default()
        };
        let (lats, lons) = grid_latlon(&grid);
        assert_eq!(lats.len(), 4);
        // First point should be close to (20, -100)
        assert!((lats[0] - 20.0).abs() < 0.01, "lat[0]={}", lats[0]);
        assert!((lons[0] - (-100.0)).abs() < 0.01, "lon[0]={}", lons[0]);
    }
}

/// Template 3.90: Space View Perspective (satellite imagery).
///
/// Simplified inverse projection for geostationary satellite imagery.
fn space_view_grid(grid: &GridDefinition, nx: usize, ny: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lats = Vec::with_capacity(n);
    let mut lons = Vec::with_capacity(n);

    let rad2deg = 180.0 / std::f64::consts::PI;

    let r_earth = 6_371_229.0_f64;
    let h = grid.altitude + r_earth;

    let cfac = if grid.dx > 0.0 { grid.dx } else { 1.0 };
    let lfac = if grid.dy > 0.0 { grid.dy } else { 1.0 };

    let xp = grid.xp;
    let yp = grid.yp;

    for j in 0..ny {
        for i in 0..nx {
            let x = (i as f64 - xp) / cfac;
            let y = (j as f64 - yp) / lfac;

            let cos_x = x.cos();
            let cos_y = y.cos();
            let sin_x = x.sin();
            let sin_y = y.sin();

            let a = sin_x * sin_x
                + cos_x * cos_x * (cos_y * cos_y + (r_earth / h).powi(2) * sin_y * sin_y);
            let sd = (h * cos_x * cos_y).powi(2)
                - (cos_x * cos_x + (h / r_earth).powi(2) * sin_x * sin_x)
                    * (h * h - r_earth * r_earth);

            if sd < 0.0 {
                lats.push(f64::NAN);
                lons.push(f64::NAN);
                continue;
            }

            let sn = (h * cos_x * cos_y - sd.sqrt()) / a;

            let s1 = h - sn * cos_x * cos_y;
            let s2 = sn * sin_x * cos_y;
            let s3 = -sn * sin_y;

            let sxy = (s1 * s1 + s2 * s2).sqrt();

            let lat = (s3 / sxy).atan() * rad2deg;
            let mut lon = (s2 / s1).atan() * rad2deg + grid.satellite_lon;

            if lon > 360.0 {
                lon -= 360.0;
            } else if lon < -180.0 {
                lon += 360.0;
            }

            lats.push(lat);
            lons.push(lon);
        }
    }

    (lats, lons)
}
