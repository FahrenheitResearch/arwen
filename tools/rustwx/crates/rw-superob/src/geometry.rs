//! Radar beam geometry, transcribed from `gpuwm/obs/geometry.py`.
//!
//! The effective-earth ("4/3-earth") model: a straight ray over a sphere of
//! radius `k * a`.  Every expression keeps the reference's operation order
//! (left-associative products and sums), because the gate's cell is
//! `rint` of a projection of these numbers and a reordered sum is a
//! different last bit.

use crate::num::{clip, py_mod};

/// `np.pi / 180.0`.
pub const DEG: f64 = std::f64::consts::PI / 180.0;

/// The radar antenna.
#[derive(Debug, Clone, Copy)]
pub struct Site {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_m: f64,
}

/// Where one gate is, and the beam unit vector at it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gate {
    pub lat: f64,
    pub lon: f64,
    pub height: f64,
    pub east: f64,
    pub north: f64,
    pub up: f64,
}

/// Everything about one radial that does not depend on range.  Computing
/// these once per radial is bit-identical to the reference's per-gate
/// evaluation: the same function of the same argument.
#[derive(Debug, Clone, Copy)]
pub struct Radial {
    az_deg: f64,
    el_deg: f64,
    sin_el: f64,
    cos_el: f64,
    cos_az_rad: f64,
    sin_az_rad: f64,
}

/// Site-wide constants.
#[derive(Debug, Clone, Copy)]
pub struct Propagation {
    site: Site,
    earth_radius_m: f64,
    effective_radius: f64,
    radius: f64,
    lat0: f64,
    lon0: f64,
    sin_lat0: f64,
    cos_lat0: f64,
}

impl Propagation {
    pub fn new(site: Site, earth_radius_m: f64, refraction_factor: f64) -> Self {
        // effective_earth_radius_m: float(earth_radius_m) * float(k)
        let effective_radius = earth_radius_m * refraction_factor;
        let radius = effective_radius + site.alt_m;
        let lat0 = site.lat_deg * DEG;
        let lon0 = site.lon_deg * DEG;
        Self {
            site,
            earth_radius_m,
            effective_radius,
            radius,
            lat0,
            lon0,
            sin_lat0: lat0.sin(),
            cos_lat0: lat0.cos(),
        }
    }

    pub fn radial(&self, azimuth_deg: f64, elevation_deg: f64) -> Radial {
        let el = elevation_deg * DEG;
        let az = azimuth_deg * DEG;
        Radial {
            az_deg: azimuth_deg,
            el_deg: elevation_deg,
            sin_el: el.sin(),
            cos_el: el.cos(),
            cos_az_rad: az.cos(),
            sin_az_rad: az.sin(),
        }
    }

    /// `gate_locations` for one gate.
    pub fn gate(&self, radial: &Radial, slant_range_m: f64) -> Gate {
        let r = slant_range_m;
        let radius = self.radius;
        // --- beam_geometry ---
        let slant_from_centre =
            (r * r + radius * radius + 2.0 * r * radius * radial.sin_el).sqrt();
        let height_above_antenna = slant_from_centre - radius;
        let height_msl = height_above_antenna + self.site.alt_m;
        let centre_angle = clip(r * radial.cos_el / slant_from_centre, -1.0, 1.0).asin();
        let arc = centre_angle * self.effective_radius;
        let cos_local = clip(radius * radial.cos_el / slant_from_centre, -1.0, 1.0);
        let mut local_el = cos_local.acos() / DEG;
        if radial.el_deg < 0.0 {
            local_el = -local_el;
        }
        // --- great_circle_point ---
        let delta = arc / self.earth_radius_m;
        let (sin_delta, cos_delta) = (delta.sin(), delta.cos());
        let sin_lat = clip(
            self.sin_lat0 * cos_delta + self.cos_lat0 * sin_delta * radial.cos_az_rad,
            -1.0,
            1.0,
        );
        let lat_rad = sin_lat.asin();
        let lon_rad = self.lon0
            + (radial.sin_az_rad * sin_delta * self.cos_lat0)
                .atan2(cos_delta - self.sin_lat0 * sin_lat);
        let lat_deg = lat_rad / DEG;
        let lon_deg = py_mod(lon_rad / DEG + 180.0, 360.0) - 180.0;
        // --- forward_azimuth ---
        let gate_azimuth = if arc > 0.0 {
            let lat1 = self.lat0;
            let lon1 = self.lon0;
            let lat2 = lat_deg * DEG;
            let lon2 = lon_deg * DEG;
            let dlon = lon2 - lon1;
            let back = ((-dlon).sin() * lat1.cos())
                .atan2(lat2.cos() * lat1.sin() - lat2.sin() * lat1.cos() * dlon.cos());
            py_mod(back / DEG + 180.0, 360.0)
        } else {
            radial.az_deg
        };
        // --- beam_unit_vector ---
        let cos_local_el = (local_el * DEG).cos();
        Gate {
            lat: lat_deg,
            lon: lon_deg,
            height: height_msl,
            east: (gate_azimuth * DEG).sin() * cos_local_el,
            north: (gate_azimuth * DEG).cos() * cos_local_el,
            up: (local_el * DEG).sin(),
        }
    }
}
