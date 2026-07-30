/// Map projections for meteorological grids.
/// Supports Lambert Conformal Conic (WRF, HRRR, NAM), Lat/Lon (GFS),
/// Polar Stereographic (HRRR-Alaska), Mercator, and Gaussian (ERA5).
/// All projections implement the `Projection` trait for generic rendering.

use std::f64::consts::PI;

const DEG_TO_RAD: f64 = PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / PI;

/// WRF uses a spherical earth with radius 6370 km.
const WRF_EARTH_RADIUS: f64 = 6_370_000.0;

/// GRIB2 standard uses a spherical earth with radius 6371.229 km.
const GRIB2_EARTH_RADIUS: f64 = 6_371_229.0;

/// Default earth radius (WRF convention, matches original wx-core).
const DEFAULT_EARTH_RADIUS: f64 = WRF_EARTH_RADIUS;

/// Trait for map projections used by the rendering engine.
/// Implementations must be thread-safe for parallel rendering.
pub trait Projection: Send + Sync + std::fmt::Debug {
    /// Convert (lat, lon) in degrees to grid coordinates (i, j).
    fn latlon_to_grid(&self, lat: f64, lon: f64) -> (f64, f64);

    /// Convert grid coordinates (i, j) to (lat, lon) in degrees.
    fn grid_to_latlon(&self, i: f64, j: f64) -> (f64, f64);

    /// Bounding box in degrees: (min_lat, min_lon, max_lat, max_lon).
    fn bounding_box(&self) -> (f64, f64, f64, f64);

    /// Number of grid points in x (west-east) direction.
    fn nx(&self) -> u32;

    /// Number of grid points in y (south-north) direction.
    fn ny(&self) -> u32;
}

// ============================================================
// Lambert Conformal Conic Projection
// ============================================================

#[derive(Debug, Clone)]
pub struct LambertProjection {
    pub latin1: f64,  // radians
    pub latin2: f64,  // radians
    pub lov: f64,     // radians (STAND_LON)
    pub la1: f64,     // radians (first grid point lat)
    pub lo1: f64,     // radians (first grid point lon)
    pub dx: f64,
    pub dy: f64,
    nx_val: u32,
    ny_val: u32,
    // Derived
    n: f64,
    f_val: f64,
    rho1: f64,
    theta1: f64,
    earth_radius: f64,
}

impl LambertProjection {
    /// Create a Lambert projection with explicit earth radius.
    pub fn with_earth_radius(
        latin1_deg: f64, latin2_deg: f64, lov_deg: f64,
        la1_deg: f64, lo1_deg: f64,
        dx: f64, dy: f64, nx: u32, ny: u32,
        earth_radius: f64,
    ) -> Self {
        let latin1 = latin1_deg * DEG_TO_RAD;
        let latin2 = latin2_deg * DEG_TO_RAD;
        let lov = lov_deg * DEG_TO_RAD;
        let la1 = la1_deg * DEG_TO_RAD;
        let lo1 = lo1_deg * DEG_TO_RAD;

        let n = if (latin1 - latin2).abs() < 1e-10 {
            latin1.sin()
        } else {
            let ln_ratio = ((PI / 4.0 + latin2 / 2.0).tan().ln()) - ((PI / 4.0 + latin1 / 2.0).tan().ln());
            (latin1.cos().ln() - latin2.cos().ln()) / ln_ratio
        };

        let f_val = (latin1.cos() * (PI / 4.0 + latin1 / 2.0).tan().powf(n)) / n;
        let rho1 = earth_radius * f_val / (PI / 4.0 + la1 / 2.0).tan().powf(n);
        let theta1 = n * (lo1 - lov);

        Self {
            latin1, latin2, lov, la1, lo1, dx, dy,
            nx_val: nx, ny_val: ny, n, f_val, rho1, theta1,
            earth_radius,
        }
    }

    /// Create with the default earth radius (6,370,000 m, WRF convention).
    pub fn new(
        latin1_deg: f64, latin2_deg: f64, lov_deg: f64,
        la1_deg: f64, lo1_deg: f64,
        dx: f64, dy: f64, nx: u32, ny: u32,
    ) -> Self {
        Self::with_earth_radius(latin1_deg, latin2_deg, lov_deg, la1_deg, lo1_deg, dx, dy, nx, ny, DEFAULT_EARTH_RADIUS)
    }

    /// Convenience constructor for WRF grids (earth radius = 6,370,000 m).
    pub fn wrf(
        latin1_deg: f64, latin2_deg: f64, lov_deg: f64,
        la1_deg: f64, lo1_deg: f64,
        dx: f64, dy: f64, nx: u32, ny: u32,
    ) -> Self {
        Self::with_earth_radius(latin1_deg, latin2_deg, lov_deg, la1_deg, lo1_deg, dx, dy, nx, ny, WRF_EARTH_RADIUS)
    }

    /// Convenience constructor for GRIB2 grids (earth radius = 6,371,229 m).
    pub fn grib2(
        latin1_deg: f64, latin2_deg: f64, lov_deg: f64,
        la1_deg: f64, lo1_deg: f64,
        dx: f64, dy: f64, nx: u32, ny: u32,
    ) -> Self {
        Self::with_earth_radius(latin1_deg, latin2_deg, lov_deg, la1_deg, lo1_deg, dx, dy, nx, ny, GRIB2_EARTH_RADIUS)
    }

    /// Build from WRF global attributes (center lat/lon based).
    /// WRF gives CEN_LAT/CEN_LON for the domain center; this computes the
    /// southwest corner (la1, lo1) needed by the projection math.
    pub fn from_wrf(truelat1: f64, truelat2: f64, stand_lon: f64, cen_lat: f64, cen_lon: f64, dx: f64, dy: f64, nx: u32, ny: u32) -> Self {
        Self::from_wrf_with_earth_radius(truelat1, truelat2, stand_lon, cen_lat, cen_lon, dx, dy, nx, ny, WRF_EARTH_RADIUS)
    }

    /// Build from WRF global attributes with explicit earth radius.
    pub fn from_wrf_with_earth_radius(
        truelat1: f64, truelat2: f64, stand_lon: f64,
        cen_lat: f64, cen_lon: f64,
        dx: f64, dy: f64, nx: u32, ny: u32,
        earth_radius: f64,
    ) -> Self {
        let latin1 = truelat1 * DEG_TO_RAD;
        let latin2 = truelat2 * DEG_TO_RAD;
        let lov = stand_lon * DEG_TO_RAD;

        let n = if (latin1 - latin2).abs() < 1e-10 {
            latin1.sin()
        } else {
            let ln_ratio = ((PI / 4.0 + latin2 / 2.0).tan().ln()) - ((PI / 4.0 + latin1 / 2.0).tan().ln());
            (latin1.cos().ln() - latin2.cos().ln()) / ln_ratio
        };

        let f_val = (latin1.cos() * (PI / 4.0 + latin1 / 2.0).tan().powf(n)) / n;

        // Project center point to Lambert x/y
        let cen_lat_r = cen_lat * DEG_TO_RAD;
        let cen_lon_r = cen_lon * DEG_TO_RAD;
        let rho_cen = earth_radius * f_val / (PI / 4.0 + cen_lat_r / 2.0).tan().powf(n);
        let theta_cen = n * (cen_lon_r - lov);
        let cx = rho_cen * theta_cen.sin();
        let cy = -rho_cen * theta_cen.cos();

        // Corner (0,0) is at center minus half-domain
        let x0 = cx - (nx as f64 - 1.0) / 2.0 * dx;
        let y0 = cy - (ny as f64 - 1.0) / 2.0 * dy;

        // Inverse project corner to get la1, lo1
        let rho_ref = earth_radius * f_val;
        let theta0 = x0.atan2(-y0);
        let rho0_abs = (x0 * x0 + y0 * y0).sqrt();

        let la1_r = 2.0 * (rho_ref / rho0_abs).powf(1.0 / n).atan() - PI / 2.0;
        let lo1_r = lov + theta0 / n;

        let la1_deg = la1_r * RAD_TO_DEG;
        let lo1_deg = lo1_r * RAD_TO_DEG;

        Self::with_earth_radius(truelat1, truelat2, stand_lon, la1_deg, lo1_deg, dx, dy, nx, ny, earth_radius)
    }

    /// Returns the earth radius used by this projection.
    pub fn earth_radius(&self) -> f64 {
        self.earth_radius
    }
}

impl Projection for LambertProjection {
    fn latlon_to_grid(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        let lat = lat_deg * DEG_TO_RAD;
        let lon = lon_deg * DEG_TO_RAD;
        let rho = self.earth_radius * self.f_val / (PI / 4.0 + lat / 2.0).tan().powf(self.n);
        let theta = self.n * (lon - self.lov);
        let x = rho * theta.sin() - self.rho1 * self.theta1.sin();
        let y = self.rho1 * self.theta1.cos() - rho * theta.cos();
        (x / self.dx, y / self.dy)
    }

    fn grid_to_latlon(&self, i: f64, j: f64) -> (f64, f64) {
        let x = self.rho1 * self.theta1.sin() + i * self.dx;
        let y = self.rho1 * self.theta1.cos() - j * self.dy;
        let rho = (x * x + y * y).sqrt() * self.n.signum();
        let theta = x.atan2(y);
        let lat = (2.0 * ((self.earth_radius * self.f_val / rho).powf(1.0 / self.n)).atan() - PI / 2.0) * RAD_TO_DEG;
        let mut lon = (self.lov + theta / self.n) * RAD_TO_DEG;
        while lon > 180.0 { lon -= 360.0; }
        while lon < -180.0 { lon += 360.0; }
        (lat, lon)
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let corners = [
            self.grid_to_latlon(0.0, 0.0),
            self.grid_to_latlon(self.nx_val as f64 - 1.0, 0.0),
            self.grid_to_latlon(0.0, self.ny_val as f64 - 1.0),
            self.grid_to_latlon(self.nx_val as f64 - 1.0, self.ny_val as f64 - 1.0),
        ];
        let min_lat = corners.iter().map(|c| c.0).fold(f64::MAX, f64::min);
        let max_lat = corners.iter().map(|c| c.0).fold(f64::MIN, f64::max);
        let min_lon = corners.iter().map(|c| c.1).fold(f64::MAX, f64::min);
        let max_lon = corners.iter().map(|c| c.1).fold(f64::MIN, f64::max);
        (min_lat, min_lon, max_lat, max_lon)
    }

    fn nx(&self) -> u32 { self.nx_val }
    fn ny(&self) -> u32 { self.ny_val }
}

// ============================================================
// Lat/Lon (Equidistant Cylindrical) Projection
// ============================================================

/// Simple equidistant cylindrical (plate carree) projection.
/// Grid index maps directly to lat/lon coordinates.
#[derive(Debug, Clone)]
pub struct LatLonProjection {
    /// Southwest corner latitude (degrees)
    pub lat1: f64,
    /// Southwest corner longitude (degrees)
    pub lon1: f64,
    /// Northeast corner latitude (degrees)
    pub lat2: f64,
    /// Northeast corner longitude (degrees)
    pub lon2: f64,
    /// Number of grid points in longitude direction
    nx_val: u32,
    /// Number of grid points in latitude direction
    ny_val: u32,
    /// Grid spacing in latitude (degrees)
    pub dlat: f64,
    /// Grid spacing in longitude (degrees)
    pub dlon: f64,
}

impl LatLonProjection {
    pub fn new(lat1: f64, lon1: f64, lat2: f64, lon2: f64, nx: u32, ny: u32) -> Self {
        let dlat = if ny > 1 { (lat2 - lat1) / (ny - 1) as f64 } else { 1.0 };
        let dlon = if nx > 1 { (lon2 - lon1) / (nx - 1) as f64 } else { 1.0 };
        Self { lat1, lon1, lat2, lon2, nx_val: nx, ny_val: ny, dlat, dlon }
    }

    /// Build from GFS-style grid definition.
    /// lat1/lon1 = first grid point, lat2/lon2 = last grid point.
    pub fn from_gfs(lat1: f64, lon1: f64, lat2: f64, lon2: f64, nx: u32, ny: u32) -> Self {
        Self::new(lat1, lon1, lat2, lon2, nx, ny)
    }
}

impl Projection for LatLonProjection {
    fn latlon_to_grid(&self, lat: f64, lon: f64) -> (f64, f64) {
        let i = (lon - self.lon1) / self.dlon;
        let j = (lat - self.lat1) / self.dlat;
        (i, j)
    }

    fn grid_to_latlon(&self, i: f64, j: f64) -> (f64, f64) {
        let lat = self.lat1 + j * self.dlat;
        let lon = self.lon1 + i * self.dlon;
        (lat, lon)
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let min_lat = self.lat1.min(self.lat2);
        let max_lat = self.lat1.max(self.lat2);
        let min_lon = self.lon1.min(self.lon2);
        let max_lon = self.lon1.max(self.lon2);
        (min_lat, min_lon, max_lat, max_lon)
    }

    fn nx(&self) -> u32 { self.nx_val }
    fn ny(&self) -> u32 { self.ny_val }
}

// ============================================================
// Polar Stereographic Projection
// ============================================================

/// Polar stereographic projection used by HRRR-Alaska, NWPS, and Canadian models.
#[derive(Debug, Clone)]
pub struct PolarStereoProjection {
    pub lov: f64,     // radians -- orientation longitude
    pub lad: f64,     // radians -- true latitude
    pub la1: f64,     // radians -- first grid point latitude
    pub lo1: f64,     // radians -- first grid point longitude
    pub dx: f64,
    pub dy: f64,
    nx_val: u32,
    ny_val: u32,
    pub south_pole: bool,
    // Derived
    k: f64,
    x0: f64,
    y0: f64,
    earth_radius: f64,
}

impl PolarStereoProjection {
    /// Create with explicit earth radius.
    pub fn with_earth_radius(
        lov_deg: f64, lad_deg: f64, la1_deg: f64, lo1_deg: f64,
        dx: f64, dy: f64, nx: u32, ny: u32, south_pole: bool,
        earth_radius: f64,
    ) -> Self {
        let lov = lov_deg * DEG_TO_RAD;
        let lad = lad_deg * DEG_TO_RAD;
        let la1 = la1_deg * DEG_TO_RAD;
        let lo1 = lo1_deg * DEG_TO_RAD;

        let k = (1.0 + lad.abs().sin()) / 2.0;

        let (x0, y0) = if south_pole {
            let t = (PI / 4.0 + la1 / 2.0).tan();
            let rho = 2.0 * earth_radius * k * t;
            let theta = lo1 - lov;
            (rho * theta.sin(), rho * theta.cos())
        } else {
            let t = (PI / 4.0 - la1 / 2.0).tan();
            let rho = 2.0 * earth_radius * k * t;
            let theta = lo1 - lov;
            (rho * theta.sin(), -rho * theta.cos())
        };

        Self { lov, lad, la1, lo1, dx, dy, nx_val: nx, ny_val: ny, south_pole, k, x0, y0, earth_radius }
    }

    pub fn new(
        lov_deg: f64, lad_deg: f64, la1_deg: f64, lo1_deg: f64,
        dx: f64, dy: f64, nx: u32, ny: u32, south_pole: bool,
    ) -> Self {
        Self::with_earth_radius(lov_deg, lad_deg, la1_deg, lo1_deg, dx, dy, nx, ny, south_pole, DEFAULT_EARTH_RADIUS)
    }

    /// Returns the earth radius used by this projection.
    pub fn earth_radius(&self) -> f64 {
        self.earth_radius
    }
}

impl Projection for PolarStereoProjection {
    fn latlon_to_grid(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        let lat = lat_deg * DEG_TO_RAD;
        let lon = lon_deg * DEG_TO_RAD;

        let (x, y) = if self.south_pole {
            let t = (PI / 4.0 + lat / 2.0).tan();
            let rho = 2.0 * self.earth_radius * self.k * t;
            let theta = lon - self.lov;
            (rho * theta.sin(), rho * theta.cos())
        } else {
            let t = (PI / 4.0 - lat / 2.0).tan();
            let rho = 2.0 * self.earth_radius * self.k * t;
            let theta = lon - self.lov;
            (rho * theta.sin(), -rho * theta.cos())
        };

        ((x - self.x0) / self.dx, (y - self.y0) / self.dy)
    }

    fn grid_to_latlon(&self, i: f64, j: f64) -> (f64, f64) {
        let x = self.x0 + i * self.dx;
        let y = self.y0 + j * self.dy;

        if self.south_pole {
            let rho = (x * x + y * y).sqrt();
            let lat = if rho.abs() < 1e-10 {
                -90.0
            } else {
                (2.0 * (rho / (2.0 * self.earth_radius * self.k)).atan() - PI / 2.0) * RAD_TO_DEG
            };
            let mut lon = (self.lov + x.atan2(y)) * RAD_TO_DEG;
            while lon > 180.0 { lon -= 360.0; }
            while lon < -180.0 { lon += 360.0; }
            (lat, lon)
        } else {
            let rho = (x * x + y * y).sqrt();
            let lat = if rho.abs() < 1e-10 {
                90.0
            } else {
                (PI / 2.0 - 2.0 * (rho / (2.0 * self.earth_radius * self.k)).atan()) * RAD_TO_DEG
            };
            let mut lon = (self.lov + x.atan2(-y)) * RAD_TO_DEG;
            while lon > 180.0 { lon -= 360.0; }
            while lon < -180.0 { lon += 360.0; }
            (lat, lon)
        }
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let corners = [
            self.grid_to_latlon(0.0, 0.0),
            self.grid_to_latlon(self.nx_val as f64 - 1.0, 0.0),
            self.grid_to_latlon(0.0, self.ny_val as f64 - 1.0),
            self.grid_to_latlon(self.nx_val as f64 - 1.0, self.ny_val as f64 - 1.0),
        ];
        let min_lat = corners.iter().map(|c| c.0).fold(f64::MAX, f64::min);
        let max_lat = corners.iter().map(|c| c.0).fold(f64::MIN, f64::max);
        let min_lon = corners.iter().map(|c| c.1).fold(f64::MAX, f64::min);
        let max_lon = corners.iter().map(|c| c.1).fold(f64::MIN, f64::max);
        (min_lat, min_lon, max_lat, max_lon)
    }

    fn nx(&self) -> u32 { self.nx_val }
    fn ny(&self) -> u32 { self.ny_val }
}

// ============================================================
// Mercator Projection
// ============================================================

/// Standard Mercator projection used by some tropical/regional models.
#[derive(Debug, Clone)]
pub struct MercatorProjection {
    pub lad: f64,     // radians -- latitude where dx/dy are true
    pub la1: f64,     // radians -- first grid point latitude
    pub lo1: f64,     // radians -- first grid point longitude
    pub dx: f64,
    pub dy: f64,
    nx_val: u32,
    ny_val: u32,
    // Derived
    cos_lad: f64,
    x0: f64,
    y0: f64,
    earth_radius: f64,
}

impl MercatorProjection {
    /// Create with explicit earth radius.
    pub fn with_earth_radius(
        lad_deg: f64, la1_deg: f64, lo1_deg: f64,
        dx: f64, dy: f64, nx: u32, ny: u32,
        earth_radius: f64,
    ) -> Self {
        let lad = lad_deg * DEG_TO_RAD;
        let la1 = la1_deg * DEG_TO_RAD;
        let lo1 = lo1_deg * DEG_TO_RAD;
        let cos_lad = lad.cos();

        let x0 = earth_radius * cos_lad * lo1;
        let y0 = earth_radius * cos_lad * ((PI / 4.0 + la1 / 2.0).tan()).ln();

        Self { lad, la1, lo1, dx, dy, nx_val: nx, ny_val: ny, cos_lad, x0, y0, earth_radius }
    }

    pub fn new(
        lad_deg: f64, la1_deg: f64, lo1_deg: f64,
        dx: f64, dy: f64, nx: u32, ny: u32,
    ) -> Self {
        Self::with_earth_radius(lad_deg, la1_deg, lo1_deg, dx, dy, nx, ny, DEFAULT_EARTH_RADIUS)
    }

    /// Returns the earth radius used by this projection.
    pub fn earth_radius(&self) -> f64 {
        self.earth_radius
    }
}

impl Projection for MercatorProjection {
    fn latlon_to_grid(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        let lat = lat_deg * DEG_TO_RAD;
        let lon = lon_deg * DEG_TO_RAD;

        let x = self.earth_radius * self.cos_lad * lon;
        let y = self.earth_radius * self.cos_lad * ((PI / 4.0 + lat / 2.0).tan()).ln();

        ((x - self.x0) / self.dx, (y - self.y0) / self.dy)
    }

    fn grid_to_latlon(&self, i: f64, j: f64) -> (f64, f64) {
        let x = self.x0 + i * self.dx;
        let y = self.y0 + j * self.dy;

        let lon = (x / (self.earth_radius * self.cos_lad)) * RAD_TO_DEG;
        let lat = (2.0 * (y / (self.earth_radius * self.cos_lad)).exp().atan() - PI / 2.0) * RAD_TO_DEG;

        (lat, lon)
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let corners = [
            self.grid_to_latlon(0.0, 0.0),
            self.grid_to_latlon(self.nx_val as f64 - 1.0, 0.0),
            self.grid_to_latlon(0.0, self.ny_val as f64 - 1.0),
            self.grid_to_latlon(self.nx_val as f64 - 1.0, self.ny_val as f64 - 1.0),
        ];
        let min_lat = corners.iter().map(|c| c.0).fold(f64::MAX, f64::min);
        let max_lat = corners.iter().map(|c| c.0).fold(f64::MIN, f64::max);
        let min_lon = corners.iter().map(|c| c.1).fold(f64::MAX, f64::min);
        let max_lon = corners.iter().map(|c| c.1).fold(f64::MIN, f64::max);
        (min_lat, min_lon, max_lat, max_lon)
    }

    fn nx(&self) -> u32 { self.nx_val }
    fn ny(&self) -> u32 { self.ny_val }
}

// ============================================================
// Gaussian Latitude/Longitude Projection
// ============================================================

/// Gaussian lat/lon projection used by ECMWF/ERA5.
/// This is an approximation that treats Gaussian latitudes as regularly spaced.
/// For exact Gaussian latitudes, roots of Legendre polynomials would be needed.
#[derive(Debug, Clone)]
pub struct GaussianProjection {
    pub lat1: f64,
    pub lon1: f64,
    pub lat2: f64,
    pub lon2: f64,
    nx_val: u32,
    ny_val: u32,
    pub dlat: f64,
    pub dlon: f64,
}

impl GaussianProjection {
    pub fn new(lat1: f64, lon1: f64, lat2: f64, lon2: f64, nx: u32, ny: u32) -> Self {
        let dlat = if ny > 1 { (lat2 - lat1) / (ny - 1) as f64 } else { 1.0 };
        let dlon = if nx > 1 { (lon2 - lon1) / (nx - 1) as f64 } else { 1.0 };
        Self { lat1, lon1, lat2, lon2, nx_val: nx, ny_val: ny, dlat, dlon }
    }
}

impl Projection for GaussianProjection {
    fn latlon_to_grid(&self, lat: f64, lon: f64) -> (f64, f64) {
        let i = (lon - self.lon1) / self.dlon;
        let j = (lat - self.lat1) / self.dlat;
        (i, j)
    }

    fn grid_to_latlon(&self, i: f64, j: f64) -> (f64, f64) {
        let lat = self.lat1 + j * self.dlat;
        let lon = self.lon1 + i * self.dlon;
        (lat, lon)
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let min_lat = self.lat1.min(self.lat2);
        let max_lat = self.lat1.max(self.lat2);
        let min_lon = self.lon1.min(self.lon2);
        let max_lon = self.lon1.max(self.lon2);
        (min_lat, min_lon, max_lat, max_lon)
    }

    fn nx(&self) -> u32 { self.nx_val }
    fn ny(&self) -> u32 { self.ny_val }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambert_roundtrip() {
        let proj = LambertProjection::new(
            33.0, 45.0, -97.0,
            21.0, -122.0,
            3000.0, 3000.0,
            500, 400,
        );

        let (lat, lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - 21.0).abs() < 0.1, "lat={}", lat);
        assert!((lon - (-122.0)).abs() < 0.1, "lon={}", lon);

        let (lat_mid, lon_mid) = proj.grid_to_latlon(250.0, 200.0);
        let (i, j) = proj.latlon_to_grid(lat_mid, lon_mid);
        assert!((i - 250.0).abs() < 0.01, "i={}", i);
        assert!((j - 200.0).abs() < 0.01, "j={}", j);
    }

    #[test]
    fn test_lambert_from_wrf() {
        let proj = LambertProjection::from_wrf(
            33.0, 45.0, -97.0,
            39.0, -97.0,
            3000.0, 3000.0,
            500, 400,
        );

        let (ci, cj) = proj.latlon_to_grid(39.0, -97.0);
        assert!((ci - 249.5).abs() < 1.0, "ci={}", ci);
        assert!((cj - 199.5).abs() < 1.0, "cj={}", cj);
    }

    #[test]
    fn test_lambert_earth_radius_constructors() {
        let wrf = LambertProjection::wrf(33.0, 45.0, -97.0, 21.0, -122.0, 3000.0, 3000.0, 500, 400);
        assert!((wrf.earth_radius() - 6_370_000.0).abs() < 0.1);

        let grib = LambertProjection::grib2(33.0, 45.0, -97.0, 21.0, -122.0, 3000.0, 3000.0, 500, 400);
        assert!((grib.earth_radius() - 6_371_229.0).abs() < 0.1);
    }

    #[test]
    fn test_latlon_roundtrip() {
        let proj = LatLonProjection::new(
            20.0, -130.0,
            55.0, -60.0,
            281, 141,
        );

        let (lat, lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - 20.0).abs() < 0.01);
        assert!((lon - (-130.0)).abs() < 0.01);

        let mid_lat = 37.5;
        let mid_lon = -95.0;
        let (i, j) = proj.latlon_to_grid(mid_lat, mid_lon);
        let (lat2, lon2) = proj.grid_to_latlon(i, j);
        assert!((lat2 - mid_lat).abs() < 0.01);
        assert!((lon2 - mid_lon).abs() < 0.01);
    }

    #[test]
    fn test_projection_trait_dyn() {
        let lambert: Box<dyn Projection> = Box::new(LambertProjection::new(
            33.0, 45.0, -97.0, 21.0, -122.0, 3000.0, 3000.0, 500, 400,
        ));
        let latlon: Box<dyn Projection> = Box::new(LatLonProjection::new(
            20.0, -130.0, 55.0, -60.0, 281, 141,
        ));

        assert_eq!(lambert.nx(), 500);
        assert_eq!(latlon.nx(), 281);

        let (lat, lon) = lambert.grid_to_latlon(0.0, 0.0);
        assert!(lat > 0.0);
        assert!(lon < 0.0);
    }

    #[test]
    fn test_polar_stereo_roundtrip() {
        let proj = PolarStereoProjection::new(
            -150.0, 60.0, 40.0, -170.0,
            3000.0, 3000.0, 200, 200, false,
        );

        let (lat, lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - 40.0).abs() < 0.1, "lat={}", lat);
        assert!((lon - (-170.0)).abs() < 0.1, "lon={}", lon);

        let (lat_mid, lon_mid) = proj.grid_to_latlon(100.0, 100.0);
        let (i, j) = proj.latlon_to_grid(lat_mid, lon_mid);
        assert!((i - 100.0).abs() < 0.01, "i={}", i);
        assert!((j - 100.0).abs() < 0.01, "j={}", j);
    }

    #[test]
    fn test_polar_stereo_south() {
        let proj = PolarStereoProjection::new(
            0.0, -60.0, -70.0, -30.0,
            5000.0, 5000.0, 100, 100, true,
        );

        let (lat, _lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - (-70.0)).abs() < 0.1, "lat={}", lat);

        let (lat2, lon2) = proj.grid_to_latlon(50.0, 50.0);
        let (i, j) = proj.latlon_to_grid(lat2, lon2);
        assert!((i - 50.0).abs() < 0.01, "i={}", i);
        assert!((j - 50.0).abs() < 0.01, "j={}", j);
    }

    #[test]
    fn test_mercator_roundtrip() {
        let proj = MercatorProjection::new(
            20.0, 10.0, -80.0,
            5000.0, 5000.0, 200, 150,
        );

        let (lat, lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - 10.0).abs() < 0.1, "lat={}", lat);
        assert!((lon - (-80.0)).abs() < 0.1, "lon={}", lon);

        let (lat_mid, lon_mid) = proj.grid_to_latlon(100.0, 75.0);
        let (i, j) = proj.latlon_to_grid(lat_mid, lon_mid);
        assert!((i - 100.0).abs() < 0.01, "i={}", i);
        assert!((j - 75.0).abs() < 0.01, "j={}", j);
    }

    #[test]
    fn test_gaussian_roundtrip() {
        let proj = GaussianProjection::new(
            90.0, 0.0, -90.0, 359.5, 720, 361,
        );

        let (lat, lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - 90.0).abs() < 0.01);
        assert!((lon - 0.0).abs() < 0.01);

        let (i, j) = proj.latlon_to_grid(0.0, 180.0);
        let (lat2, lon2) = proj.grid_to_latlon(i, j);
        assert!((lat2 - 0.0).abs() < 0.01);
        assert!((lon2 - 180.0).abs() < 0.01);
    }

    #[test]
    fn test_new_projections_trait_dyn() {
        let polar: Box<dyn Projection> = Box::new(PolarStereoProjection::new(
            -150.0, 60.0, 40.0, -170.0, 3000.0, 3000.0, 200, 200, false,
        ));
        let mercator: Box<dyn Projection> = Box::new(MercatorProjection::new(
            20.0, 10.0, -80.0, 5000.0, 5000.0, 200, 150,
        ));
        let gaussian: Box<dyn Projection> = Box::new(GaussianProjection::new(
            90.0, 0.0, -90.0, 359.5, 720, 361,
        ));

        assert_eq!(polar.nx(), 200);
        assert_eq!(mercator.nx(), 200);
        assert_eq!(gaussian.nx(), 720);
    }

    #[test]
    fn test_projections_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LambertProjection>();
        assert_send_sync::<LatLonProjection>();
        assert_send_sync::<PolarStereoProjection>();
        assert_send_sync::<MercatorProjection>();
        assert_send_sync::<GaussianProjection>();
    }
}
