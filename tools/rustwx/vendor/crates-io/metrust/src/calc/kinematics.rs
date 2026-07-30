//! Kinematics and dynamics calculations for 2D gridded meteorological fields.
//!
//! This module provides a MetPy-compatible API by re-exporting functions from
//! [`wx_math::dynamics`] and [`wx_math::gridmath`], plus new implementations
//! such as baroclinic potential vorticity.
//!
//! All grid arrays are flattened row-major: `index = j * nx + i` where `j` is the
//! y-index (row) and `i` is the x-index (column). `dx` and `dy` are grid spacings
//! in meters.

// ─────────────────────────────────────────────────────────────
// Re-exports from wx_math::dynamics with MetPy-compatible names
// ─────────────────────────────────────────────────────────────

/// Horizontal divergence: du/dx + dv/dy.
///
/// # Arguments
/// * `u` - Zonal (east-west) wind component, flattened row-major
/// * `v` - Meridional (north-south) wind component, flattened row-major
/// * `nx` - Number of grid points in x
/// * `ny` - Number of grid points in y
/// * `dx` - Grid spacing in x (meters)
/// * `dy` - Grid spacing in y (meters)
pub use wx_math::dynamics::divergence;

/// Relative vorticity: dv/dx - du/dy.
///
/// Positive values indicate cyclonic rotation (counterclockwise in the Northern
/// Hemisphere).
pub use wx_math::dynamics::vorticity;

/// Absolute vorticity: relative vorticity + Coriolis parameter.
///
/// # Arguments
/// * `u`, `v` - Wind components (flattened row-major)
/// * `lats` - Latitude in degrees at each grid point (flattened row-major)
/// * `nx`, `ny` - Grid dimensions
/// * `dx`, `dy` - Grid spacings (meters)
pub use wx_math::dynamics::absolute_vorticity;

/// Advection of a scalar field by a 2D wind: -u(ds/dx) - v(ds/dy).
///
/// Returns the local rate of change of the scalar due to advection.
pub use wx_math::dynamics::advection;

/// 2D Petterssen frontogenesis function.
///
/// Measures the rate of change of the magnitude of the potential temperature
/// gradient due to the deforming wind field. Re-exported from
/// `wx_math::dynamics::frontogenesis_2d`.
pub fn frontogenesis(
    theta: &[f64], u: &[f64], v: &[f64],
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    wx_math::dynamics::frontogenesis_2d(theta, u, v, nx, ny, dx, dy)
}

/// Geostrophic wind from a geopotential height field.
///
/// Returns `(u_geo, v_geo)` where:
/// * `u_g = -(g/f) * dZ/dy`
/// * `v_g = (g/f) * dZ/dx`
///
/// Near the equator (|f| < 1e-10) the geostrophic approximation is undefined
/// and the output is zeroed.
pub use wx_math::dynamics::geostrophic_wind;

/// Ageostrophic wind: total wind minus geostrophic wind.
///
/// Returns `(u - u_geo, v - v_geo)`.
pub use wx_math::dynamics::ageostrophic_wind;

/// Q-vector components (Q1, Q2) on a constant pressure surface.
///
/// The Q-vector is a diagnostic of quasi-geostrophic forcing for vertical
/// motion. Returns `(q1, q2)`.
///
/// # Arguments
/// * `t` - Temperature (K), flattened row-major
/// * `u_geo`, `v_geo` - Geostrophic wind components
/// * `p_hpa` - Pressure level in hPa
/// * `nx`, `ny` - Grid dimensions
/// * `dx`, `dy` - Grid spacings (meters)
pub use wx_math::dynamics::q_vector;

/// Q-vector convergence: -2 * div(Q).
///
/// Positive values indicate QG forcing for ascent.
pub use wx_math::dynamics::q_vector_convergence;

/// Stretching deformation: du/dx - dv/dy.
pub use wx_math::dynamics::stretching_deformation;

/// Shearing deformation: dv/dx + du/dy.
pub use wx_math::dynamics::shearing_deformation;

/// Total deformation: sqrt(stretching^2 + shearing^2).
pub use wx_math::dynamics::total_deformation;

/// Coriolis parameter: f = 2 * Omega * sin(latitude).
///
/// # Arguments
/// * `lat_deg` - Latitude in degrees
pub use wx_math::dynamics::coriolis_parameter;

// ─────────────────────────────────────────────────────────────
// Re-exports from wx_math::dynamics (additional utilities)
// ─────────────────────────────────────────────────────────────

/// Partial derivative df/dx using centered finite differences (forward/backward
/// at boundaries).
pub use wx_math::dynamics::gradient_x;

/// Partial derivative df/dy using centered finite differences (forward/backward
/// at boundaries).
pub use wx_math::dynamics::gradient_y;

/// Laplacian: d2f/dx2 + d2f/dy2.
pub use wx_math::dynamics::laplacian;

/// Scalar wind speed: sqrt(u^2 + v^2).
pub use wx_math::dynamics::wind_speed;

/// Meteorological wind direction (degrees, 0 = from north, 90 = from east).
pub use wx_math::dynamics::wind_direction;

/// Convert wind speed and meteorological direction to (u, v) components.
pub use wx_math::dynamics::wind_components;

/// Temperature advection (convenience wrapper around `advection`).
pub use wx_math::dynamics::temperature_advection;

/// Moisture advection (convenience wrapper around `advection`).
pub use wx_math::dynamics::moisture_advection;

/// Curvature vorticity -- the component of vorticity from streamline curvature.
pub use wx_math::dynamics::curvature_vorticity;

/// Shear vorticity -- the component of vorticity from cross-stream speed shear.
pub use wx_math::dynamics::shear_vorticity;

/// Inertial-advective wind: advection of the geostrophic wind by the total wind.
pub use wx_math::dynamics::inertial_advective_wind;

/// Absolute momentum: M = u - f * y.
pub use wx_math::dynamics::absolute_momentum;

/// Kinematic flux: element-wise product of a velocity component and a scalar.
pub use wx_math::dynamics::kinematic_flux;

// ─────────────────────────────────────────────────────────────
// Re-exports from wx_math::gridmath
// ─────────────────────────────────────────────────────────────

/// Generalized first derivative along a chosen axis (0 = x, 1 = y).
pub use wx_math::gridmath::first_derivative;

/// Generalized second derivative along a chosen axis (0 = x, 1 = y).
pub use wx_math::gridmath::second_derivative;

/// Compute physical grid spacings (dx, dy) in meters from lat/lon arrays
/// using the haversine formula.
pub use wx_math::gridmath::lat_lon_grid_deltas;

// ─────────────────────────────────────────────────────────────
// New: Baroclinic potential vorticity (Ertel PV)
// ─────────────────────────────────────────────────────────────

/// Standard gravitational acceleration (m/s^2).
const G: f64 = 9.80665;

/// Baroclinic (Ertel) potential vorticity on a 2D isobaric slice.
///
/// Computes a simplified form of Ertel's PV suitable for a single
/// pressure level:
///
/// ```text
/// PV = -g * (f + zeta) * (d_theta / d_p)
/// ```
///
/// where `f + zeta` is the absolute vorticity (Coriolis parameter plus
/// relative vorticity), and `d_theta / d_p` is the static stability
/// approximated by finite differences between adjacent pressure levels.
///
/// # Arguments
///
/// * `potential_temp` - Potential temperature (K) on the *current* level,
///   flattened row-major, length `nx * ny`.
/// * `pressure` - Two-element slice `[p_below, p_above]` in Pa, representing
///   the pressure levels that bracket the current level.
/// * `theta_below` - Potential temperature on the level below (higher
///   pressure), flattened row-major, length `nx * ny`.
/// * `theta_above` - Potential temperature on the level above (lower
///   pressure), flattened row-major, length `nx * ny`.
/// * `u`, `v` - Wind components on the current level (m/s), each length
///   `nx * ny`.
/// * `lats` - Latitude in degrees at each grid point, length `nx * ny`.
/// * `nx`, `ny` - Grid dimensions.
/// * `dx`, `dy` - Grid spacings in meters.
///
/// # Returns
///
/// A `Vec<f64>` of length `nx * ny` containing PV in PVU-like units
/// (K m^2 kg^-1 s^-1). Multiply by 1e6 to obtain standard PVU.
///
/// # Panics
///
/// Panics if any input array has an unexpected length or if the two pressure
/// levels are equal.
///
/// # Example
///
/// ```
/// use metrust::calc::kinematics::potential_vorticity_baroclinic;
///
/// let nx = 3;
/// let ny = 3;
/// let n = nx * ny;
/// // Uniform theta on current level, small gradient across levels
/// let theta = vec![300.0; n];
/// let theta_below = vec![298.0; n];
/// let theta_above = vec![302.0; n];
/// let pressure = [85000.0, 70000.0]; // 850 hPa and 700 hPa in Pa
/// let u = vec![10.0; n];
/// let v = vec![5.0; n];
/// let lats = vec![45.0; n];
/// let dx = 50_000.0;
/// let dy = 50_000.0;
///
/// let pv = potential_vorticity_baroclinic(
///     &theta, &pressure, &theta_below, &theta_above,
///     &u, &v, &lats, nx, ny, dx, dy,
/// );
/// assert_eq!(pv.len(), n);
/// ```
pub fn potential_vorticity_baroclinic(
    potential_temp: &[f64],
    pressure: &[f64; 2],
    theta_below: &[f64],
    theta_above: &[f64],
    u: &[f64],
    v: &[f64],
    lats: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(potential_temp.len(), n, "potential_temp length mismatch");
    assert_eq!(theta_below.len(), n, "theta_below length mismatch");
    assert_eq!(theta_above.len(), n, "theta_above length mismatch");
    assert_eq!(u.len(), n, "u length mismatch");
    assert_eq!(v.len(), n, "v length mismatch");
    assert_eq!(lats.len(), n, "lats length mismatch");

    let p_below = pressure[0];
    let p_above = pressure[1];
    let dp = p_above - p_below;
    assert!(
        dp.abs() > 1e-10,
        "pressure levels must be different (got dp = {})",
        dp
    );

    // Absolute vorticity: f + zeta
    let abs_vort = wx_math::dynamics::absolute_vorticity(u, v, lats, nx, ny, dx, dy);

    // Static stability: d_theta / d_p via finite difference between the two
    // bracketing levels.
    //
    // In a standard atmosphere theta increases with decreasing pressure, so
    // d_theta/d_p is normally negative. We can optionally use the current-level
    // theta for refinement, but the simplest correct form uses the two
    // bracketing levels.
    let mut pv = vec![0.0; n];
    for k in 0..n {
        let dthetadp = (theta_above[k] - theta_below[k]) / dp;
        pv[k] = -G * abs_vort[k] * dthetadp;
    }

    pv
}

// ─────────────────────────────────────────────────────────────
// Barotropic potential vorticity
// ─────────────────────────────────────────────────────────────

/// Barotropic potential vorticity: absolute vorticity divided by layer
/// depth.
///
/// ```text
/// PV_bt = (f + zeta) / h
/// ```
///
/// # Arguments
///
/// * `heights` - Layer depth or height field (m), flattened row-major,
///   length `nx * ny`.
/// * `u`, `v` - Wind components (m/s), each length `nx * ny`.
/// * `lats` - Latitude in degrees at each grid point, length `nx * ny`.
/// * `nx`, `ny` - Grid dimensions.
/// * `dx`, `dy` - Grid spacings in meters.
///
/// # Returns
///
/// A `Vec<f64>` of length `nx * ny` containing barotropic PV
/// (s^-1 m^-1).  Where `heights` is zero the output is set to
/// `f64::NAN` to avoid division by zero.
pub fn potential_vorticity_barotropic(
    heights: &[f64],
    u: &[f64],
    v: &[f64],
    lats: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(heights.len(), n, "heights length mismatch");
    assert_eq!(u.len(), n, "u length mismatch");
    assert_eq!(v.len(), n, "v length mismatch");
    assert_eq!(lats.len(), n, "lats length mismatch");

    let abs_vort = wx_math::dynamics::absolute_vorticity(u, v, lats, nx, ny, dx, dy);

    let mut pv = vec![0.0; n];
    for k in 0..n {
        if heights[k].abs() < 1e-30 {
            pv[k] = f64::NAN;
        } else {
            pv[k] = abs_vort[k] / heights[k];
        }
    }
    pv
}

// ─────────────────────────────────────────────────────────────
// Cross-section wind decomposition
// ─────────────────────────────────────────────────────────────

/// Decompose wind into components parallel and perpendicular to a
/// cross-section line.
///
/// The cross-section is defined by `start` and `end` as `(lat, lon)` in
/// degrees.  The "parallel" component is positive in the direction from
/// start to end; the "perpendicular" component is positive 90 degrees
/// counterclockwise from the parallel direction.
///
/// # Arguments
///
/// * `u` - Zonal wind component (m/s).
/// * `v` - Meridional wind component (m/s).
/// * `start` - `(lat, lon)` of the cross-section start in degrees.
/// * `end` - `(lat, lon)` of the cross-section end in degrees.
///
/// # Returns
///
/// `(parallel, perpendicular)` each the same length as `u` / `v`.
pub fn cross_section_components(
    u: &[f64],
    v: &[f64],
    start: (f64, f64),
    end: (f64, f64),
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(u.len(), v.len(), "u and v must have the same length");

    let to_rad = std::f64::consts::PI / 180.0;

    // Approximate unit vector along the cross-section in (east, north)
    // space.  We project the great-circle bearing into local Cartesian
    // offsets.  For a straight cross-section this gives a constant
    // direction.
    let dlat = (end.0 - start.0) * to_rad;
    let dlon = (end.1 - start.1) * to_rad;
    let mean_lat = ((start.0 + end.0) / 2.0) * to_rad;

    let de = dlon * mean_lat.cos(); // east displacement (radians)
    let dn = dlat;                   // north displacement (radians)
    let mag = (de * de + dn * dn).sqrt();

    if mag < 1e-30 {
        // Degenerate: start == end, return zeros.
        let z = vec![0.0; u.len()];
        return (z.clone(), z);
    }

    // Unit vector along the section (east, north components).
    let ue = de / mag;
    let un = dn / mag;

    let n = u.len();
    let mut parallel = Vec::with_capacity(n);
    let mut perpendicular = Vec::with_capacity(n);

    for i in 0..n {
        // u is east, v is north
        let par = u[i] * ue + v[i] * un;
        let perp = -u[i] * un + v[i] * ue;
        parallel.push(par);
        perpendicular.push(perp);
    }

    (parallel, perpendicular)
}

// ─────────────────────────────────────────────────────────────
// Unit vectors for cross-section decomposition
// ─────────────────────────────────────────────────────────────

/// Compute tangent and normal unit vectors for a cross-section line.
///
/// Given a cross-section defined by `start` and `end` as `(lat, lon)` in
/// degrees, returns `(tangent, normal)` where each is an `(east, north)`
/// unit vector pair.
///
/// The tangent vector points from `start` toward `end`. The normal
/// vector is 90 degrees counterclockwise from the tangent (i.e.,
/// `normal = (-tangent_north, tangent_east)`).
///
/// If start and end are the same point the returned vectors are `(0, 0)`.
///
/// # Example
///
/// ```
/// use metrust::calc::kinematics::unit_vectors_from_cross_section;
///
/// let (tang, norm) = unit_vectors_from_cross_section((40.0, -100.0), (40.0, -80.0));
/// // East-west section: tangent is purely eastward
/// assert!((tang.0 - 1.0).abs() < 1e-10);
/// assert!(tang.1.abs() < 1e-10);
/// ```
pub fn unit_vectors_from_cross_section(
    start: (f64, f64),
    end: (f64, f64),
) -> ((f64, f64), (f64, f64)) {
    let to_rad = std::f64::consts::PI / 180.0;

    let dlat = (end.0 - start.0) * to_rad;
    let dlon = (end.1 - start.1) * to_rad;
    let mean_lat = ((start.0 + end.0) / 2.0) * to_rad;

    let de = dlon * mean_lat.cos(); // east displacement (radians)
    let dn = dlat;                   // north displacement (radians)
    let mag = (de * de + dn * dn).sqrt();

    if mag < 1e-30 {
        return ((0.0, 0.0), (0.0, 0.0));
    }

    let te = de / mag; // tangent east component
    let tn = dn / mag; // tangent north component

    // Normal: 90 degrees counterclockwise from tangent
    let ne = -tn;
    let nn = te;

    ((te, tn), (ne, nn))
}

// ─────────────────────────────────────────────────────────────
// Tangential and normal wind components
// ─────────────────────────────────────────────────────────────

/// Component of wind tangential (parallel) to a cross-section line.
///
/// The cross-section is defined by `start` and `end` as `(lat, lon)` in
/// degrees. The tangential component is the dot product of the wind
/// vector `(u, v)` with the unit tangent vector pointing from start
/// to end.
///
/// # Arguments
///
/// * `u` - Zonal wind component (m/s).
/// * `v` - Meridional wind component (m/s).
/// * `start` - `(lat, lon)` of the cross-section start in degrees.
/// * `end` - `(lat, lon)` of the cross-section end in degrees.
///
/// # Example
///
/// ```
/// use metrust::calc::kinematics::tangential_component;
///
/// let u = vec![10.0];
/// let v = vec![0.0];
/// // Due-east section: tangential = u
/// let tang = tangential_component(&u, &v, (45.0, -100.0), (45.0, -80.0));
/// assert!((tang[0] - 10.0).abs() < 1e-10);
/// ```
pub fn tangential_component(
    u: &[f64],
    v: &[f64],
    start: (f64, f64),
    end: (f64, f64),
) -> Vec<f64> {
    assert_eq!(u.len(), v.len(), "u and v must have the same length");
    let ((te, tn), _) = unit_vectors_from_cross_section(start, end);
    u.iter()
        .zip(v.iter())
        .map(|(&ui, &vi)| ui * te + vi * tn)
        .collect()
}

/// Component of wind normal (perpendicular) to a cross-section line.
///
/// The cross-section is defined by `start` and `end` as `(lat, lon)` in
/// degrees. The normal component is the dot product of the wind vector
/// `(u, v)` with the unit normal vector, which is 90 degrees
/// counterclockwise from the tangent direction.
///
/// # Arguments
///
/// * `u` - Zonal wind component (m/s).
/// * `v` - Meridional wind component (m/s).
/// * `start` - `(lat, lon)` of the cross-section start in degrees.
/// * `end` - `(lat, lon)` of the cross-section end in degrees.
///
/// # Example
///
/// ```
/// use metrust::calc::kinematics::normal_component;
///
/// let u = vec![0.0];
/// let v = vec![10.0];
/// // Due-east section: normal = v
/// let norm = normal_component(&u, &v, (45.0, -100.0), (45.0, -80.0));
/// assert!((norm[0] - 10.0).abs() < 1e-10);
/// ```
pub fn normal_component(
    u: &[f64],
    v: &[f64],
    start: (f64, f64),
    end: (f64, f64),
) -> Vec<f64> {
    assert_eq!(u.len(), v.len(), "u and v must have the same length");
    let (_, (ne, nn)) = unit_vectors_from_cross_section(start, end);
    u.iter()
        .zip(v.iter())
        .map(|(&ui, &vi)| ui * ne + vi * nn)
        .collect()
}

// ─────────────────────────────────────────────────────────────
// Velocity gradient tensor
// ─────────────────────────────────────────────────────────────

/// Compute all four components of the 2D velocity gradient tensor.
///
/// Returns `(du_dx, du_dy, dv_dx, dv_dy)` using centered finite
/// differences in the interior and forward/backward differences at the
/// boundaries.
///
/// # Arguments
///
/// * `u` - Zonal wind component, flattened row-major, length `nx * ny`.
/// * `v` - Meridional wind component, flattened row-major, length `nx * ny`.
/// * `nx` - Number of grid points in x.
/// * `ny` - Number of grid points in y.
/// * `dx` - Grid spacing in x (meters).
/// * `dy` - Grid spacing in y (meters).
///
/// # Returns
///
/// `(du_dx, du_dy, dv_dx, dv_dy)` -- each a `Vec<f64>` of length `nx * ny`.
///
/// # Example
///
/// ```
/// use metrust::calc::kinematics::vector_derivative;
///
/// // u = x, v = -y  =>  du/dx=1, du/dy=0, dv/dx=0, dv/dy=-1
/// let nx = 5;
/// let ny = 5;
/// let n = nx * ny;
/// let mut u = vec![0.0; n];
/// let mut v = vec![0.0; n];
/// for j in 0..ny {
///     for i in 0..nx {
///         u[j * nx + i] = i as f64;
///         v[j * nx + i] = -(j as f64);
///     }
/// }
/// let (du_dx, du_dy, dv_dx, dv_dy) = vector_derivative(&u, &v, nx, ny, 1.0, 1.0);
/// // Interior point
/// assert!((du_dx[12] - 1.0).abs() < 1e-10);
/// assert!((dv_dy[12] + 1.0).abs() < 1e-10);
/// ```
pub fn vector_derivative(
    u: &[f64],
    v: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = nx * ny;
    assert_eq!(u.len(), n, "u length must equal nx * ny");
    assert_eq!(v.len(), n, "v length must equal nx * ny");

    let du_dx = wx_math::dynamics::gradient_x(u, nx, ny, dx);
    let du_dy = wx_math::dynamics::gradient_y(u, nx, ny, dy);
    let dv_dx = wx_math::dynamics::gradient_x(v, nx, ny, dx);
    let dv_dy = wx_math::dynamics::gradient_y(v, nx, ny, dy);

    (du_dx, du_dy, dv_dx, dv_dy)
}

// ─────────────────────────────────────────────────────────────
// 3D advection
// ─────────────────────────────────────────────────────────────

/// Advection of a scalar field by a 3-D wind: -u(ds/dx) - v(ds/dy) - w(ds/dz).
///
/// Extends the 2-D advection to include the vertical advection term.  The
/// scalar, u, v, and w fields are 3-D arrays flattened in level-major
/// order: `index = k * ny * nx + j * nx + i`.
///
/// `dz` is the spacing between vertical levels in meters.
///
/// # Arguments
///
/// * `scalar` - 3-D scalar field, flattened `[nz * ny * nx]`
/// * `u` - Zonal wind component, flattened `[nz * ny * nx]`
/// * `v` - Meridional wind component, flattened `[nz * ny * nx]`
/// * `w` - Vertical velocity, flattened `[nz * ny * nx]`
/// * `nx`, `ny`, `nz` - Grid dimensions
/// * `dx`, `dy` - Horizontal grid spacings (meters)
/// * `dz` - Vertical grid spacing (meters)
pub fn advection_3d(
    scalar: &[f64],
    u: &[f64],
    v: &[f64],
    w: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
) -> Vec<f64> {
    let nxy = nx * ny;
    let n = nxy * nz;
    assert_eq!(scalar.len(), n, "scalar length mismatch");
    assert_eq!(u.len(), n, "u length mismatch");
    assert_eq!(v.len(), n, "v length mismatch");
    assert_eq!(w.len(), n, "w length mismatch");

    let mut out = vec![0.0; n];

    for k in 0..nz {
        let offset = k * nxy;
        let slab_s = &scalar[offset..offset + nxy];
        let slab_u = &u[offset..offset + nxy];
        let slab_v = &v[offset..offset + nxy];
        let slab_w = &w[offset..offset + nxy];

        // Horizontal gradients for this level.
        let dsdx = wx_math::dynamics::gradient_x(slab_s, nx, ny, dx);
        let dsdy = wx_math::dynamics::gradient_y(slab_s, nx, ny, dy);

        for ij in 0..nxy {
            let idx = offset + ij;
            // Horizontal advection: -u ds/dx - v ds/dy.
            out[idx] = -slab_u[ij] * dsdx[ij] - slab_v[ij] * dsdy[ij];

            // Vertical advection: -w ds/dz (centered differences).
            let dsdz = if nz < 2 {
                0.0
            } else if k == 0 {
                (scalar[(k + 1) * nxy + ij] - scalar[k * nxy + ij]) / dz
            } else if k == nz - 1 {
                (scalar[k * nxy + ij] - scalar[(k - 1) * nxy + ij]) / dz
            } else {
                (scalar[(k + 1) * nxy + ij] - scalar[(k - 1) * nxy + ij]) / (2.0 * dz)
            };
            out[idx] -= slab_w[ij] * dsdz;
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------
    // Re-export smoke tests: verify the wrappers compile and
    // produce correct results on trivial inputs.
    // ---------------------------------------------------------

    #[test]
    fn test_divergence_uniform() {
        let n = 4 * 3;
        let u = vec![5.0; n];
        let v = vec![3.0; n];
        let div = divergence(&u, &v, 4, 3, 1000.0, 1000.0);
        for val in &div {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_vorticity_solid_rotation() {
        // u = -y, v = x => zeta = 2
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = -(j as f64);
                v[j * nx + i] = i as f64;
            }
        }
        let vort = vorticity(&u, &v, nx, ny, 1.0, 1.0);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    (vort[j * nx + i] - 2.0).abs() < 1e-10,
                    "vorticity at ({},{}) = {}, expected 2.0",
                    i, j, vort[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_advection_linear() {
        // scalar = x, u = 1, v = 0 => advection = -1
        let nx = 5;
        let ny = 3;
        let n = nx * ny;
        let mut scalar = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                scalar[j * nx + i] = i as f64;
            }
        }
        let u = vec![1.0; n];
        let v = vec![0.0; n];
        let adv = advection(&scalar, &u, &v, nx, ny, 1.0, 1.0);
        for j in 0..ny {
            for i in 1..nx - 1 {
                assert!(
                    (adv[j * nx + i] + 1.0).abs() < 1e-10,
                    "advection at ({},{}) = {}, expected -1.0",
                    i, j, adv[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_frontogenesis_wrapper() {
        // Constant theta => zero frontogenesis
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let theta = vec![300.0; n];
        let u = vec![10.0; n];
        let v = vec![5.0; n];
        let fg = frontogenesis(&theta, &u, &v, nx, ny, 1000.0, 1000.0);
        for val in &fg {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_coriolis_parameter_values() {
        let f_45 = coriolis_parameter(45.0);
        let expected = 2.0 * 7.2921159e-5 * (45.0_f64 * std::f64::consts::PI / 180.0).sin();
        assert!((f_45 - expected).abs() < 1e-12);

        assert!(coriolis_parameter(0.0).abs() < 1e-15);
    }

    #[test]
    fn test_wind_speed_components_roundtrip() {
        let speed = vec![10.0, 20.0, 15.0];
        let direction = vec![180.0, 270.0, 45.0];
        let (u, v) = wind_components(&speed, &direction);
        let spd = wind_speed(&u, &v);
        for k in 0..3 {
            assert!((spd[k] - speed[k]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_deformation() {
        // u = x, v = -y => stretching = 2, shearing = 0, total = 2
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = i as f64;
                v[j * nx + i] = -(j as f64);
            }
        }
        let st = stretching_deformation(&u, &v, nx, ny, 1.0, 1.0);
        let sh = shearing_deformation(&u, &v, nx, ny, 1.0, 1.0);
        let td = total_deformation(&u, &v, nx, ny, 1.0, 1.0);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                assert!((st[k] - 2.0).abs() < 1e-10);
                assert!(sh[k].abs() < 1e-10);
                assert!((td[k] - 2.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_geostrophic_wind_zonal_height_gradient() {
        // Height increases linearly to the north: Z = j * 10 m
        // dZ/dy = 10/dy, dZ/dx = 0
        // u_g = -(g/f) * dZ/dy, v_g = 0
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let dx = 100_000.0;
        let dy = 100_000.0;
        let mut height = vec![0.0; n];
        let lats = vec![45.0; n];
        for j in 0..ny {
            for i in 0..nx {
                height[j * nx + i] = (j as f64) * 10.0;
            }
        }
        let (ug, vg) = geostrophic_wind(&height, &lats, nx, ny, dx, dy);
        let f = coriolis_parameter(45.0);
        let expected_ug = -(G / f) * (10.0 / dy);
        // Interior check
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                assert!(
                    (ug[k] - expected_ug).abs() < 1.0,
                    "u_geo at ({},{}) = {}, expected {}",
                    i, j, ug[k], expected_ug
                );
                assert!(vg[k].abs() < 1e-6, "v_geo should be ~0, got {}", vg[k]);
            }
        }
    }

    #[test]
    fn test_ageostrophic_wind_residual() {
        let u = vec![15.0, 20.0, 10.0];
        let v = vec![5.0, -3.0, 8.0];
        let ug = vec![12.0, 18.0, 9.0];
        let vg = vec![4.0, -2.0, 7.0];
        let (ua, va) = ageostrophic_wind(&u, &v, &ug, &vg);
        assert!((ua[0] - 3.0).abs() < 1e-10);
        assert!((ua[1] - 2.0).abs() < 1e-10);
        assert!((ua[2] - 1.0).abs() < 1e-10);
        assert!((va[0] - 1.0).abs() < 1e-10);
        assert!((va[1] + 1.0).abs() < 1e-10);
        assert!((va[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_q_vector_smoke() {
        // Uniform T => gradients zero => Q = 0
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let t = vec![280.0; n];
        let ug = vec![10.0; n];
        let vg = vec![5.0; n];
        let (q1, q2) = q_vector(&t, &ug, &vg, 500.0, nx, ny, 50_000.0, 50_000.0);
        for k in 0..n {
            assert!(q1[k].abs() < 1e-15);
            assert!(q2[k].abs() < 1e-15);
        }
    }

    #[test]
    fn test_q_vector_convergence_smoke() {
        let n = 9;
        let q1 = vec![0.0; n];
        let q2 = vec![0.0; n];
        let qc = q_vector_convergence(&q1, &q2, 3, 3, 1000.0, 1000.0);
        for val in &qc {
            assert!(val.abs() < 1e-15);
        }
    }

    #[test]
    fn test_temperature_advection_is_advection() {
        let nx = 5;
        let ny = 3;
        let n = nx * ny;
        let mut t = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                t[j * nx + i] = 280.0 + i as f64;
            }
        }
        let u = vec![2.0; n];
        let v = vec![0.0; n];
        let ta = temperature_advection(&t, &u, &v, nx, ny, 1.0, 1.0);
        let a = advection(&t, &u, &v, nx, ny, 1.0, 1.0);
        for k in 0..n {
            assert!((ta[k] - a[k]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_moisture_advection_is_advection() {
        let nx = 4;
        let ny = 4;
        let n = nx * ny;
        let mut q = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                q[j * nx + i] = 0.01 + 0.001 * j as f64;
            }
        }
        let u = vec![0.0; n];
        let v = vec![3.0; n];
        let ma = moisture_advection(&q, &u, &v, nx, ny, 1.0, 1.0);
        let a = advection(&q, &u, &v, nx, ny, 1.0, 1.0);
        for k in 0..n {
            assert!((ma[k] - a[k]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_curvature_shear_vorticity_sum() {
        // curvature + shear = total relative vorticity
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = (i as f64) * 2.0 - (j as f64);
                v[j * nx + i] = (j as f64) + (i as f64) * 0.5;
            }
        }
        let total = vorticity(&u, &v, nx, ny, 1.0, 1.0);
        let curv = curvature_vorticity(&u, &v, nx, ny, 1.0, 1.0);
        let shear = shear_vorticity(&u, &v, nx, ny, 1.0, 1.0);
        for k in 0..n {
            assert!(
                (curv[k] + shear[k] - total[k]).abs() < 1e-10,
                "curv + shear != total at k={}: {} + {} != {}",
                k, curv[k], shear[k], total[k]
            );
        }
    }

    #[test]
    fn test_kinematic_flux_elementwise() {
        let vel = vec![1.0, 2.0, 3.0, 4.0];
        let scalar = vec![10.0, 20.0, 30.0, 40.0];
        let flux = kinematic_flux(&vel, &scalar);
        assert_eq!(flux, vec![10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn test_absolute_momentum_calculation() {
        let u = vec![10.0, 20.0];
        let lats = vec![45.0, 45.0];
        let y_dist = vec![0.0, 100_000.0];
        let m = absolute_momentum(&u, &lats, &y_dist);
        let f = coriolis_parameter(45.0);
        assert!((m[0] - 10.0).abs() < 1e-10);
        let expected_1 = 20.0 - f * 100_000.0;
        assert!((m[1] - expected_1).abs() < 1e-6);
    }

    // ---------------------------------------------------------
    // Re-exports from gridmath
    // ---------------------------------------------------------

    #[test]
    fn test_first_derivative_x_linear() {
        // f = 3*i, dx = 1 => df/dx = 3
        let nx = 5;
        let ny = 3;
        let n = nx * ny;
        let mut vals = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                vals[j * nx + i] = 3.0 * i as f64;
            }
        }
        let d = first_derivative(&vals, 1.0, 0, nx, ny);
        for j in 0..ny {
            for i in 0..nx {
                assert!(
                    (d[j * nx + i] - 3.0).abs() < 1e-10,
                    "first_derivative at ({},{}) = {}, expected 3.0",
                    i, j, d[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_second_derivative_quadratic() {
        // f = i^2, dx = 1 => d2f/dx2 = 2
        let nx = 5;
        let ny = 3;
        let n = nx * ny;
        let mut vals = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                vals[j * nx + i] = (i * i) as f64;
            }
        }
        let d2 = second_derivative(&vals, 1.0, 0, nx, ny);
        for j in 0..ny {
            for i in 0..nx {
                assert!(
                    (d2[j * nx + i] - 2.0).abs() < 1e-10,
                    "second_derivative at ({},{}) = {}, expected 2.0",
                    i, j, d2[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_lat_lon_grid_deltas_smoke() {
        let nx = 3;
        let ny = 3;
        let mut lats = vec![0.0; 9];
        let mut lons = vec![0.0; 9];
        for j in 0..ny {
            for i in 0..nx {
                lats[j * nx + i] = 44.0 + j as f64;
                lons[j * nx + i] = -90.0 + i as f64;
            }
        }
        let (dx, dy) = lat_lon_grid_deltas(&lats, &lons, nx, ny);
        // At 45N, 1 degree latitude ~ 111 km
        assert!((dy[4] - 111_130.0).abs() < 500.0);
        // 1 degree longitude at 45N ~ 78.6 km
        assert!((dx[4] - 78_600.0).abs() < 1500.0);
    }

    // ---------------------------------------------------------
    // Potential vorticity tests
    // ---------------------------------------------------------

    #[test]
    fn test_pv_baroclinic_uniform_wind() {
        // Uniform wind => zero relative vorticity => PV dominated by f * dtheta/dp
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let theta = vec![300.0; n];
        let theta_below = vec![298.0; n]; // lower level (higher pressure)
        let theta_above = vec![303.0; n]; // upper level (lower pressure)
        let pressure = [85000.0, 50000.0]; // 850 hPa, 500 hPa in Pa
        let u = vec![10.0; n];
        let v = vec![5.0; n];
        let lats = vec![45.0; n];
        let dx = 100_000.0;
        let dy = 100_000.0;

        let pv = potential_vorticity_baroclinic(
            &theta, &pressure, &theta_below, &theta_above,
            &u, &v, &lats, nx, ny, dx, dy,
        );
        assert_eq!(pv.len(), n);

        // With uniform wind, relative vorticity = 0, so absolute vorticity = f
        let f = coriolis_parameter(45.0);
        let dp = pressure[1] - pressure[0]; // 50000 - 85000 = -35000
        let dtheta_dp = (303.0 - 298.0) / dp; // 5 / (-35000)
        let expected_pv = -G * f * dtheta_dp;

        // Interior points (boundary finite differences may differ slightly)
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                assert!(
                    (pv[k] - expected_pv).abs() < 1e-10,
                    "PV at ({},{}) = {}, expected {}",
                    i, j, pv[k], expected_pv
                );
            }
        }

        // PV should be positive in the Northern Hemisphere with typical stability
        // f > 0 at 45N, dp < 0, dtheta > 0, so dtheta/dp < 0
        // PV = -g * f * (negative) = positive
        assert!(expected_pv > 0.0, "Expected positive PV in NH, got {}", expected_pv);
    }

    #[test]
    fn test_pv_baroclinic_with_vorticity() {
        // Solid-body rotation contributes additional vorticity
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let theta = vec![300.0; n];
        let theta_below = vec![296.0; n];
        let theta_above = vec![304.0; n];
        let pressure = [90000.0, 70000.0]; // 900 hPa, 700 hPa
        let lats = vec![45.0; n];
        let dx = 100_000.0;
        let dy = 100_000.0;

        // u = -y, v = x => relative vorticity = 2 / (dx units)
        // scaled to physical coords: u = -j*dy, v = i*dx
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = -(j as f64) * dy;
                v[j * nx + i] = (i as f64) * dx;
            }
        }

        let pv = potential_vorticity_baroclinic(
            &theta, &pressure, &theta_below, &theta_above,
            &u, &v, &lats, nx, ny, dx, dy,
        );

        // At interior points, relative vorticity should be 2 (dv/dx - du/dy = 1 - (-1) = 2)
        // absolute vorticity = f + 2
        let f = coriolis_parameter(45.0);
        let dp = pressure[1] - pressure[0]; // -20000
        let dtheta_dp = (304.0 - 296.0) / dp; // 8 / (-20000)
        let expected = -G * (f + 2.0) * dtheta_dp;

        // Check a well-interior point
        let k = 3 * nx + 3;
        assert!(
            (pv[k] - expected).abs() / expected.abs() < 1e-6,
            "PV with rotation at center = {}, expected {}",
            pv[k], expected
        );
    }

    #[test]
    fn test_pv_baroclinic_zero_stability() {
        // If theta_above == theta_below, dtheta/dp = 0 => PV = 0
        let nx = 3;
        let ny = 3;
        let n = nx * ny;
        let theta = vec![300.0; n];
        let theta_below = vec![300.0; n];
        let theta_above = vec![300.0; n];
        let pressure = [85000.0, 50000.0];
        let u = vec![10.0; n];
        let v = vec![0.0; n];
        let lats = vec![40.0; n];

        let pv = potential_vorticity_baroclinic(
            &theta, &pressure, &theta_below, &theta_above,
            &u, &v, &lats, nx, ny, 100_000.0, 100_000.0,
        );
        for val in &pv {
            assert!(val.abs() < 1e-15, "PV should be zero with neutral stability, got {}", val);
        }
    }

    #[test]
    #[should_panic(expected = "pressure levels must be different")]
    fn test_pv_baroclinic_same_pressure_panics() {
        let n = 4;
        let theta = vec![300.0; n];
        let pressure = [85000.0, 85000.0];
        let u = vec![0.0; n];
        let v = vec![0.0; n];
        let lats = vec![45.0; n];
        let _ = potential_vorticity_baroclinic(
            &theta, &pressure, &theta, &theta,
            &u, &v, &lats, 2, 2, 1000.0, 1000.0,
        );
    }

    #[test]
    fn test_pv_baroclinic_southern_hemisphere() {
        // In the SH, f < 0 and typical PV < 0
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let theta = vec![300.0; n];
        let theta_below = vec![298.0; n];
        let theta_above = vec![303.0; n];
        let pressure = [85000.0, 50000.0];
        let u = vec![10.0; n];
        let v = vec![5.0; n];
        let lats = vec![-45.0; n];
        let dx = 100_000.0;
        let dy = 100_000.0;

        let pv = potential_vorticity_baroclinic(
            &theta, &pressure, &theta_below, &theta_above,
            &u, &v, &lats, nx, ny, dx, dy,
        );

        // f < 0 at -45, dtheta/dp < 0 => -g * (negative) * (negative) = -g * positive < 0
        let f = coriolis_parameter(-45.0);
        assert!(f < 0.0);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                assert!(
                    pv[k] < 0.0,
                    "PV in SH should be negative, got {} at ({},{})",
                    pv[k], i, j
                );
            }
        }
    }

    // ---------------------------------------------------------
    // Barotropic potential vorticity tests
    // ---------------------------------------------------------

    #[test]
    fn test_pv_barotropic_uniform_wind() {
        // Uniform wind => zero relative vorticity => PV = f / h
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let heights = vec![5000.0; n];
        let u = vec![10.0; n];
        let v = vec![5.0; n];
        let lats = vec![45.0; n];
        let dx = 100_000.0;
        let dy = 100_000.0;

        let pv = potential_vorticity_barotropic(&heights, &u, &v, &lats, nx, ny, dx, dy);
        assert_eq!(pv.len(), n);

        let f = coriolis_parameter(45.0);
        let expected = f / 5000.0;
        // Interior points
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                assert!(
                    (pv[k] - expected).abs() < 1e-15,
                    "barotropic PV at ({},{}) = {}, expected {}",
                    i, j, pv[k], expected
                );
            }
        }
    }

    #[test]
    fn test_pv_barotropic_zero_height_gives_nan() {
        let nx = 3;
        let ny = 3;
        let n = nx * ny;
        let mut heights = vec![5000.0; n];
        heights[4] = 0.0; // center point
        let u = vec![10.0; n];
        let v = vec![0.0; n];
        let lats = vec![45.0; n];

        let pv = potential_vorticity_barotropic(&heights, &u, &v, &lats, nx, ny, 1000.0, 1000.0);
        assert!(pv[4].is_nan(), "Expected NaN for zero height, got {}", pv[4]);
        // Other points should be finite
        assert!(pv[0].is_finite());
    }

    #[test]
    fn test_pv_barotropic_with_vorticity() {
        // Solid body rotation: u = -y, v = x => zeta = 2
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let heights = vec![1000.0; n];
        let lats = vec![45.0; n];
        let dx = 1.0;
        let dy = 1.0;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = -(j as f64);
                v[j * nx + i] = i as f64;
            }
        }

        let pv = potential_vorticity_barotropic(&heights, &u, &v, &lats, nx, ny, dx, dy);
        let f = coriolis_parameter(45.0);
        let expected = (f + 2.0) / 1000.0;

        // Check interior
        let k = 3 * nx + 3;
        assert!(
            (pv[k] - expected).abs() < 1e-12,
            "barotropic PV with rotation = {}, expected {}",
            pv[k], expected
        );
    }

    // ---------------------------------------------------------
    // Cross-section component tests
    // ---------------------------------------------------------

    #[test]
    fn test_cross_section_components_eastward() {
        // Cross-section runs due east => parallel = u, perpendicular = v
        let u = vec![10.0, 20.0];
        let v = vec![3.0, -5.0];
        let start = (45.0, -100.0);
        let end = (45.0, -80.0);
        let (par, perp) = cross_section_components(&u, &v, start, end);
        for i in 0..2 {
            assert!(
                (par[i] - u[i]).abs() < 1e-10,
                "parallel should match u: {} vs {}",
                par[i], u[i]
            );
            assert!(
                (perp[i] - v[i]).abs() < 1e-10,
                "perp should match v: {} vs {}",
                perp[i], v[i]
            );
        }
    }

    #[test]
    fn test_cross_section_components_northward() {
        // Cross-section runs due north => parallel = v, perpendicular = -u
        let u = vec![10.0];
        let v = vec![5.0];
        let start = (30.0, -90.0);
        let end = (50.0, -90.0);
        let (par, perp) = cross_section_components(&u, &v, start, end);
        assert!(
            (par[0] - 5.0).abs() < 1e-10,
            "parallel should be v for northward section: {}",
            par[0]
        );
        assert!(
            (perp[0] + 10.0).abs() < 1e-10,
            "perp should be -u for northward section: {}",
            perp[0]
        );
    }

    #[test]
    fn test_cross_section_components_magnitude_preserved() {
        // Rotation preserves magnitude: par^2 + perp^2 = u^2 + v^2
        let u = vec![7.0, -3.0, 15.0];
        let v = vec![4.0, 12.0, -8.0];
        let start = (35.0, -95.0);
        let end = (45.0, -80.0);
        let (par, perp) = cross_section_components(&u, &v, start, end);
        for i in 0..3 {
            let orig_sq = u[i] * u[i] + v[i] * v[i];
            let rot_sq = par[i] * par[i] + perp[i] * perp[i];
            assert!(
                (orig_sq - rot_sq).abs() < 1e-10,
                "Magnitude not preserved at {}: {} vs {}",
                i, orig_sq, rot_sq
            );
        }
    }

    #[test]
    fn test_cross_section_components_degenerate() {
        // Same start and end => zero output
        let u = vec![10.0];
        let v = vec![5.0];
        let pt = (45.0, -90.0);
        let (par, perp) = cross_section_components(&u, &v, pt, pt);
        assert!((par[0]).abs() < 1e-10);
        assert!((perp[0]).abs() < 1e-10);
    }

    // ---------------------------------------------------------
    // Unit vectors from cross section tests
    // ---------------------------------------------------------

    #[test]
    fn test_unit_vectors_eastward() {
        let (tang, norm) = unit_vectors_from_cross_section((45.0, -100.0), (45.0, -80.0));
        // Tangent should be purely eastward: (1, 0)
        assert!((tang.0 - 1.0).abs() < 1e-10, "tangent east = {}", tang.0);
        assert!(tang.1.abs() < 1e-10, "tangent north = {}", tang.1);
        // Normal 90 CCW from east = north: (0, 1)
        assert!(norm.0.abs() < 1e-10, "normal east = {}", norm.0);
        assert!((norm.1 - 1.0).abs() < 1e-10, "normal north = {}", norm.1);
    }

    #[test]
    fn test_unit_vectors_northward() {
        let (tang, norm) = unit_vectors_from_cross_section((30.0, -90.0), (50.0, -90.0));
        // Tangent should be purely northward: (0, 1)
        assert!(tang.0.abs() < 1e-10, "tangent east = {}", tang.0);
        assert!((tang.1 - 1.0).abs() < 1e-10, "tangent north = {}", tang.1);
        // Normal 90 CCW from north = west: (-1, 0)
        assert!((norm.0 + 1.0).abs() < 1e-10, "normal east = {}", norm.0);
        assert!(norm.1.abs() < 1e-10, "normal north = {}", norm.1);
    }

    #[test]
    fn test_unit_vectors_are_unit_length() {
        let (tang, norm) = unit_vectors_from_cross_section((35.0, -95.0), (45.0, -80.0));
        let t_mag = (tang.0 * tang.0 + tang.1 * tang.1).sqrt();
        let n_mag = (norm.0 * norm.0 + norm.1 * norm.1).sqrt();
        assert!((t_mag - 1.0).abs() < 1e-10, "tangent magnitude = {}", t_mag);
        assert!((n_mag - 1.0).abs() < 1e-10, "normal magnitude = {}", n_mag);
    }

    #[test]
    fn test_unit_vectors_perpendicular() {
        let (tang, norm) = unit_vectors_from_cross_section((35.0, -95.0), (45.0, -80.0));
        let dot = tang.0 * norm.0 + tang.1 * norm.1;
        assert!(dot.abs() < 1e-10, "tangent and normal must be perpendicular, dot = {}", dot);
    }

    #[test]
    fn test_unit_vectors_degenerate() {
        let (tang, norm) = unit_vectors_from_cross_section((45.0, -90.0), (45.0, -90.0));
        assert!(tang.0.abs() < 1e-10);
        assert!(tang.1.abs() < 1e-10);
        assert!(norm.0.abs() < 1e-10);
        assert!(norm.1.abs() < 1e-10);
    }

    // ---------------------------------------------------------
    // Tangential and normal component tests
    // ---------------------------------------------------------

    #[test]
    fn test_tangential_eastward_section() {
        // Due-east section: tangential = u
        let u = vec![10.0, 20.0];
        let v = vec![3.0, -5.0];
        let tang = tangential_component(&u, &v, (45.0, -100.0), (45.0, -80.0));
        for i in 0..2 {
            assert!((tang[i] - u[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normal_eastward_section() {
        // Due-east section: normal = v
        let u = vec![10.0, 20.0];
        let v = vec![3.0, -5.0];
        let norm = normal_component(&u, &v, (45.0, -100.0), (45.0, -80.0));
        for i in 0..2 {
            assert!((norm[i] - v[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tangential_northward_section() {
        // Due-north section: tangential = v
        let u = vec![10.0];
        let v = vec![5.0];
        let tang = tangential_component(&u, &v, (30.0, -90.0), (50.0, -90.0));
        assert!((tang[0] - 5.0).abs() < 1e-10, "tangential = {}", tang[0]);
    }

    #[test]
    fn test_normal_northward_section() {
        // Due-north section: normal (CCW from north = west direction) = -u
        let u = vec![10.0];
        let v = vec![5.0];
        let norm = normal_component(&u, &v, (30.0, -90.0), (50.0, -90.0));
        assert!((norm[0] + 10.0).abs() < 1e-10, "normal = {}", norm[0]);
    }

    #[test]
    fn test_tangential_normal_magnitude_preserved() {
        // tang^2 + norm^2 = u^2 + v^2
        let u = vec![7.0, -3.0, 15.0];
        let v = vec![4.0, 12.0, -8.0];
        let start = (35.0, -95.0);
        let end = (45.0, -80.0);
        let tang = tangential_component(&u, &v, start, end);
        let norm = normal_component(&u, &v, start, end);
        for i in 0..3 {
            let orig_sq = u[i] * u[i] + v[i] * v[i];
            let decomp_sq = tang[i] * tang[i] + norm[i] * norm[i];
            assert!(
                (orig_sq - decomp_sq).abs() < 1e-10,
                "magnitude not preserved at {}: {} vs {}",
                i, orig_sq, decomp_sq
            );
        }
    }

    #[test]
    fn test_tangential_normal_match_cross_section_components() {
        // tangential_component should match the parallel output of cross_section_components,
        // and normal_component should match the perpendicular output.
        let u = vec![7.0, -3.0, 15.0];
        let v = vec![4.0, 12.0, -8.0];
        let start = (35.0, -95.0);
        let end = (45.0, -80.0);
        let (par, perp) = cross_section_components(&u, &v, start, end);
        let tang = tangential_component(&u, &v, start, end);
        let norm = normal_component(&u, &v, start, end);
        for i in 0..3 {
            assert!(
                (tang[i] - par[i]).abs() < 1e-10,
                "tangential vs parallel mismatch at {}: {} vs {}",
                i, tang[i], par[i]
            );
            assert!(
                (norm[i] - perp[i]).abs() < 1e-10,
                "normal vs perpendicular mismatch at {}: {} vs {}",
                i, norm[i], perp[i]
            );
        }
    }

    // ---------------------------------------------------------
    // Vector derivative tests
    // ---------------------------------------------------------

    #[test]
    fn test_vector_derivative_linear_field() {
        // u = x, v = -y => du/dx=1, du/dy=0, dv/dx=0, dv/dy=-1
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = i as f64;
                v[j * nx + i] = -(j as f64);
            }
        }
        let (du_dx, du_dy, dv_dx, dv_dy) = vector_derivative(&u, &v, nx, ny, 1.0, 1.0);
        // Check interior points
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                assert!((du_dx[k] - 1.0).abs() < 1e-10, "du/dx = {}", du_dx[k]);
                assert!(du_dy[k].abs() < 1e-10, "du/dy = {}", du_dy[k]);
                assert!(dv_dx[k].abs() < 1e-10, "dv/dx = {}", dv_dx[k]);
                assert!((dv_dy[k] + 1.0).abs() < 1e-10, "dv/dy = {}", dv_dy[k]);
            }
        }
    }

    #[test]
    fn test_vector_derivative_solid_body_rotation() {
        // u = -y, v = x => du/dx=0, du/dy=-1, dv/dx=1, dv/dy=0
        // divergence = du/dx + dv/dy = 0
        // vorticity = dv/dx - du/dy = 1 - (-1) = 2
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = -(j as f64);
                v[j * nx + i] = i as f64;
            }
        }
        let (du_dx, du_dy, dv_dx, dv_dy) = vector_derivative(&u, &v, nx, ny, 1.0, 1.0);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                let div = du_dx[k] + dv_dy[k];
                let vort = dv_dx[k] - du_dy[k];
                assert!(div.abs() < 1e-10, "divergence = {}", div);
                assert!((vort - 2.0).abs() < 1e-10, "vorticity = {}", vort);
            }
        }
    }

    #[test]
    fn test_vector_derivative_consistent_with_divergence_vorticity() {
        // vector_derivative should give results consistent with divergence and vorticity
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = (i as f64) * 2.0 - (j as f64);
                v[j * nx + i] = (j as f64) + (i as f64) * 0.5;
            }
        }
        let dx = 1000.0;
        let dy = 1000.0;
        let (du_dx, du_dy, dv_dx, dv_dy) = vector_derivative(&u, &v, nx, ny, dx, dy);
        let div = divergence(&u, &v, nx, ny, dx, dy);
        let vort = vorticity(&u, &v, nx, ny, dx, dy);

        for k in 0..n {
            let div_from_vd = du_dx[k] + dv_dy[k];
            let vort_from_vd = dv_dx[k] - du_dy[k];
            assert!(
                (div_from_vd - div[k]).abs() < 1e-10,
                "divergence mismatch at {}: {} vs {}",
                k, div_from_vd, div[k]
            );
            assert!(
                (vort_from_vd - vort[k]).abs() < 1e-10,
                "vorticity mismatch at {}: {} vs {}",
                k, vort_from_vd, vort[k]
            );
        }
    }

    #[test]
    fn test_vector_derivative_uniform_wind() {
        // Uniform wind => all derivatives zero
        let nx = 4;
        let ny = 4;
        let n = nx * ny;
        let u = vec![10.0; n];
        let v = vec![5.0; n];
        let (du_dx, du_dy, dv_dx, dv_dy) = vector_derivative(&u, &v, nx, ny, 1000.0, 1000.0);
        for k in 0..n {
            assert!(du_dx[k].abs() < 1e-10);
            assert!(du_dy[k].abs() < 1e-10);
            assert!(dv_dx[k].abs() < 1e-10);
            assert!(dv_dy[k].abs() < 1e-10);
        }
    }

    #[test]
    fn test_vector_derivative_deformation() {
        // u = x, v = y => stretching = du/dx - dv/dy = 1-1 = 0
        // shearing = dv/dx + du/dy = 0 + 0 = 0
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = i as f64;
                v[j * nx + i] = j as f64;
            }
        }
        let (du_dx, du_dy, dv_dx, dv_dy) = vector_derivative(&u, &v, nx, ny, 1.0, 1.0);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                let stretch = du_dx[k] - dv_dy[k];
                let shear = dv_dx[k] + du_dy[k];
                assert!(stretch.abs() < 1e-10, "stretching = {}", stretch);
                assert!(shear.abs() < 1e-10, "shearing = {}", shear);
            }
        }
    }

    // ── advection_3d ──

    #[test]
    fn test_advection_3d_uniform_field() {
        // Uniform scalar field => advection should be zero everywhere.
        let nx = 4;
        let ny = 3;
        let nz = 3;
        let n = nx * ny * nz;
        let scalar = vec![10.0; n];
        let u = vec![5.0; n];
        let v = vec![3.0; n];
        let w = vec![1.0; n];
        let result = advection_3d(&scalar, &u, &v, &w, nx, ny, nz, 1000.0, 1000.0, 500.0);
        for val in &result {
            assert!(val.abs() < 1e-10, "Expected 0, got {}", val);
        }
    }

    #[test]
    fn test_advection_3d_vertical_only() {
        // Scalar varies linearly with level: s = k * 10.0, all horizontal uniform.
        // Only w-advection term should be nonzero.
        let nx = 3;
        let ny = 3;
        let nz = 3;
        let nxy = nx * ny;
        let mut scalar = vec![0.0; nxy * nz];
        for k in 0..nz {
            for ij in 0..nxy {
                scalar[k * nxy + ij] = k as f64 * 10.0;
            }
        }
        let u = vec![0.0; nxy * nz];
        let v = vec![0.0; nxy * nz];
        let w = vec![1.0; nxy * nz];
        let dz = 100.0;
        let result = advection_3d(&scalar, &u, &v, &w, nx, ny, nz, 1000.0, 1000.0, dz);
        // ds/dz = 10 / 100 = 0.1 for interior levels.
        // advection = -w * ds/dz = -1.0 * 0.1 = -0.1 for interior.
        let k = 1; // middle level
        for ij in 0..nxy {
            let val = result[k * nxy + ij];
            assert!((val + 0.1).abs() < 1e-10, "Expected -0.1, got {} at ij={}", val, ij);
        }
    }

    #[test]
    fn test_advection_3d_horizontal_only() {
        // When w=0, 3D advection should match 2D advection for each level.
        let nx = 5;
        let ny = 5;
        let nz = 2;
        let nxy = nx * ny;
        // Linear ramp in x: s = i.
        let mut scalar = vec![0.0; nxy * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    scalar[k * nxy + j * nx + i] = i as f64;
                }
            }
        }
        let u = vec![2.0; nxy * nz];
        let v = vec![0.0; nxy * nz];
        let w = vec![0.0; nxy * nz];
        let dx = 1.0;
        let dy = 1.0;
        let dz = 1.0;
        let result_3d = advection_3d(&scalar, &u, &v, &w, nx, ny, nz, dx, dy, dz);
        // Compare with 2D advection for each level.
        for k in 0..nz {
            let slab_s = &scalar[k * nxy..(k + 1) * nxy];
            let slab_u = &u[k * nxy..(k + 1) * nxy];
            let slab_v = &v[k * nxy..(k + 1) * nxy];
            let result_2d = advection(slab_s, slab_u, slab_v, nx, ny, dx, dy);
            for ij in 0..nxy {
                assert!(
                    (result_3d[k * nxy + ij] - result_2d[ij]).abs() < 1e-10,
                    "Mismatch at k={}, ij={}: 3d={}, 2d={}",
                    k, ij, result_3d[k * nxy + ij], result_2d[ij]
                );
            }
        }
    }
}
