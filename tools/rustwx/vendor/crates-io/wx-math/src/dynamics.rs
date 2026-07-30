/// Dynamics and kinematics calculations for 2D grids.
///
/// All grid arrays are flattened row-major: index = j * nx + i
/// where j is the y-index (row) and i is the x-index (column).
/// `dx` and `dy` are grid spacings in meters.

use std::f64::consts::PI;
use rayon::prelude::*;

/// Earth's angular velocity (rad/s).
const OMEGA: f64 = 7.2921159e-5;

/// Specific gas constant for dry air (J/(kg·K)).
const RD: f64 = 287.058;

/// Gravitational acceleration (m/s²).
const G: f64 = 9.80665;

// ─────────────────────────────────────────────
// Helper: index into row-major grid
// ─────────────────────────────────────────────

#[inline(always)]
fn idx(j: usize, i: usize, nx: usize) -> usize {
    j * nx + i
}

// ─────────────────────────────────────────────
// Finite-difference derivatives
// ─────────────────────────────────────────────

/// ∂f/∂x using centered differences (2nd-order one-sided at boundaries).
pub fn gradient_x(values: &[f64], nx: usize, ny: usize, dx: f64) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny);
    let inv_2dx = 1.0 / (2.0 * dx);
    let inv_dx = 1.0 / dx;

    let rows: Vec<Vec<f64>> = (0..ny).into_par_iter().map(|j| {
        (0..nx).map(|i| {
            if nx < 2 {
                0.0
            } else if nx == 2 {
                (values[idx(j, 1, nx)] - values[idx(j, 0, nx)]) * inv_dx
            } else if i == 0 {
                (-3.0 * values[idx(j, 0, nx)] + 4.0 * values[idx(j, 1, nx)]
                    - values[idx(j, 2, nx)]) * inv_2dx
            } else if i == nx - 1 {
                (3.0 * values[idx(j, nx - 1, nx)] - 4.0 * values[idx(j, nx - 2, nx)]
                    + values[idx(j, nx - 3, nx)]) * inv_2dx
            } else {
                (values[idx(j, i + 1, nx)] - values[idx(j, i - 1, nx)]) * inv_2dx
            }
        }).collect()
    }).collect();
    rows.into_iter().flatten().collect()
}

/// ∂f/∂y using centered differences (2nd-order one-sided at boundaries).
pub fn gradient_y(values: &[f64], nx: usize, ny: usize, dy: f64) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny);
    let inv_2dy = 1.0 / (2.0 * dy);
    let inv_dy = 1.0 / dy;

    let rows: Vec<Vec<f64>> = (0..ny).into_par_iter().map(|j| {
        (0..nx).map(|i| {
            if ny < 2 {
                0.0
            } else if ny == 2 {
                (values[idx(1, i, nx)] - values[idx(0, i, nx)]) * inv_dy
            } else if j == 0 {
                (-3.0 * values[idx(0, i, nx)] + 4.0 * values[idx(1, i, nx)]
                    - values[idx(2, i, nx)]) * inv_2dy
            } else if j == ny - 1 {
                (3.0 * values[idx(ny - 1, i, nx)] - 4.0 * values[idx(ny - 2, i, nx)]
                    + values[idx(ny - 3, i, nx)]) * inv_2dy
            } else {
                (values[idx(j + 1, i, nx)] - values[idx(j - 1, i, nx)]) * inv_2dy
            }
        }).collect()
    }).collect();
    rows.into_iter().flatten().collect()
}

/// Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y².
pub fn laplacian(values: &[f64], nx: usize, ny: usize, dx: f64, dy: f64) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny);
    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);

    let rows: Vec<Vec<f64>> = (0..ny).into_par_iter().map(|j| {
        (0..nx).map(|i| {
            let d2x = if nx < 3 {
                0.0
            } else if i == 0 {
                (values[idx(j, 2, nx)] - 2.0 * values[idx(j, 1, nx)] + values[idx(j, 0, nx)]) * inv_dx2
            } else if i == nx - 1 {
                (values[idx(j, nx - 1, nx)] - 2.0 * values[idx(j, nx - 2, nx)] + values[idx(j, nx - 3, nx)]) * inv_dx2
            } else {
                (values[idx(j, i + 1, nx)] - 2.0 * values[idx(j, i, nx)] + values[idx(j, i - 1, nx)]) * inv_dx2
            };
            let d2y = if ny < 3 {
                0.0
            } else if j == 0 {
                (values[idx(2, i, nx)] - 2.0 * values[idx(1, i, nx)] + values[idx(0, i, nx)]) * inv_dy2
            } else if j == ny - 1 {
                (values[idx(ny - 1, i, nx)] - 2.0 * values[idx(ny - 2, i, nx)] + values[idx(ny - 3, i, nx)]) * inv_dy2
            } else {
                (values[idx(j + 1, i, nx)] - 2.0 * values[idx(j, i, nx)] + values[idx(j - 1, i, nx)]) * inv_dy2
            };
            d2x + d2y
        }).collect()
    }).collect();
    rows.into_iter().flatten().collect()
}

// ─────────────────────────────────────────────
// Wind dynamics
// ─────────────────────────────────────────────

/// Horizontal divergence: ∂u/∂x + ∂v/∂y.
pub fn divergence(u: &[f64], v: &[f64], nx: usize, ny: usize, dx: f64, dy: f64) -> Vec<f64> {
    let dudx = gradient_x(u, nx, ny, dx);
    let dvdy = gradient_y(v, nx, ny, dy);
    dudx.par_iter().zip(dvdy.par_iter()).map(|(a, b)| a + b).collect()
}

/// Relative vorticity: ∂v/∂x - ∂u/∂y.
pub fn vorticity(u: &[f64], v: &[f64], nx: usize, ny: usize, dx: f64, dy: f64) -> Vec<f64> {
    let dvdx = gradient_x(v, nx, ny, dx);
    let dudy = gradient_y(u, nx, ny, dy);
    dvdx.par_iter().zip(dudy.par_iter()).map(|(a, b)| a - b).collect()
}

/// Coriolis parameter: f = 2Ω sin(φ).
pub fn coriolis_parameter(lat_deg: f64) -> f64 {
    2.0 * OMEGA * (lat_deg * PI / 180.0).sin()
}

/// Absolute vorticity: relative vorticity + Coriolis parameter.
/// `lats` is a flattened array of latitudes (degrees) at each grid point.
pub fn absolute_vorticity(
    u: &[f64], v: &[f64], lats: &[f64],
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    let rel = vorticity(u, v, nx, ny, dx, dy);
    assert_eq!(lats.len(), nx * ny);
    rel.par_iter()
        .zip(lats.par_iter())
        .map(|(zeta, lat)| zeta + coriolis_parameter(*lat))
        .collect()
}

/// Stretching deformation: ∂u/∂x - ∂v/∂y.
pub fn stretching_deformation(
    u: &[f64], v: &[f64], nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    let dudx = gradient_x(u, nx, ny, dx);
    let dvdy = gradient_y(v, nx, ny, dy);
    dudx.par_iter().zip(dvdy.par_iter()).map(|(a, b)| a - b).collect()
}

/// Shearing deformation: ∂v/∂x + ∂u/∂y.
pub fn shearing_deformation(
    u: &[f64], v: &[f64], nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    let dvdx = gradient_x(v, nx, ny, dx);
    let dudy = gradient_y(u, nx, ny, dy);
    dvdx.par_iter().zip(dudy.par_iter()).map(|(a, b)| a + b).collect()
}

/// Total deformation: √(stretching² + shearing²).
pub fn total_deformation(
    u: &[f64], v: &[f64], nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    let st = stretching_deformation(u, v, nx, ny, dx, dy);
    let sh = shearing_deformation(u, v, nx, ny, dx, dy);
    st.par_iter()
        .zip(sh.par_iter())
        .map(|(s, h)| (s * s + h * h).sqrt())
        .collect()
}

// ─────────────────────────────────────────────
// Advection
// ─────────────────────────────────────────────

/// Advection of a scalar field: -u(∂s/∂x) - v(∂s/∂y).
pub fn advection(
    scalar: &[f64], u: &[f64], v: &[f64],
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    let dsdx = gradient_x(scalar, nx, ny, dx);
    let dsdy = gradient_y(scalar, nx, ny, dy);
    (0..nx * ny).into_par_iter()
        .map(|k| -u[k] * dsdx[k] - v[k] * dsdy[k])
        .collect()
}

/// Temperature advection (wrapper around `advection`).
pub fn temperature_advection(
    t: &[f64], u: &[f64], v: &[f64],
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    advection(t, u, v, nx, ny, dx, dy)
}

/// Moisture advection (wrapper around `advection`).
pub fn moisture_advection(
    q: &[f64], u: &[f64], v: &[f64],
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    advection(q, u, v, nx, ny, dx, dy)
}

// ─────────────────────────────────────────────
// Frontogenesis
// ─────────────────────────────────────────────

/// 2D Petterssen frontogenesis function:
///
/// F = -1/(|∇θ|) * [ (∂θ/∂x)²(∂u/∂x) + (∂θ/∂y)²(∂v/∂y)
///                   + (∂θ/∂x)(∂θ/∂y)(∂v/∂x + ∂u/∂y) ]
///
/// This is the rate of change of the magnitude of the potential temperature gradient.
pub fn frontogenesis_2d(
    theta: &[f64], u: &[f64], v: &[f64],
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    let dtdx = gradient_x(theta, nx, ny, dx);
    let dtdy = gradient_y(theta, nx, ny, dy);
    let dudx = gradient_x(u, nx, ny, dx);
    let dvdy = gradient_y(v, nx, ny, dy);
    let dvdx = gradient_x(v, nx, ny, dx);
    let dudy = gradient_y(u, nx, ny, dy);

    let n = nx * ny;
    let mut out = vec![0.0; n];
    for k in 0..n {
        let mag_grad = (dtdx[k] * dtdx[k] + dtdy[k] * dtdy[k]).sqrt();
        if mag_grad < 1e-20 {
            out[k] = 0.0;
        } else {
            let fg = -(dtdx[k] * dtdx[k] * dudx[k]
                + dtdy[k] * dtdy[k] * dvdy[k]
                + dtdx[k] * dtdy[k] * (dvdx[k] + dudy[k]))
                / mag_grad;
            out[k] = fg;
        }
    }
    out
}

// ─────────────────────────────────────────────
// QG Forcing
// ─────────────────────────────────────────────

/// Q-vector components (Q1, Q2) on a constant pressure surface.
///
/// Q1 = -(Rd / p) * [∂u_g/∂x · ∂T/∂x + ∂v_g/∂x · ∂T/∂y]
/// Q2 = -(Rd / p) * [∂u_g/∂y · ∂T/∂x + ∂v_g/∂y · ∂T/∂y]
///
/// `p_hpa` is the pressure level in hPa. Internally converted to Pa.
pub fn q_vector(
    t: &[f64], u_geo: &[f64], v_geo: &[f64],
    p_hpa: f64,
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> (Vec<f64>, Vec<f64>) {
    let p_pa = p_hpa * 100.0;
    let coeff = -RD / p_pa;

    let dtdx = gradient_x(t, nx, ny, dx);
    let dtdy = gradient_y(t, nx, ny, dy);
    let dugdx = gradient_x(u_geo, nx, ny, dx);
    let dugdy = gradient_y(u_geo, nx, ny, dy);
    let dvgdx = gradient_x(v_geo, nx, ny, dx);
    let dvgdy = gradient_y(v_geo, nx, ny, dy);

    let n = nx * ny;
    let mut q1 = vec![0.0; n];
    let mut q2 = vec![0.0; n];
    for k in 0..n {
        q1[k] = coeff * (dugdx[k] * dtdx[k] + dvgdx[k] * dtdy[k]);
        q2[k] = coeff * (dugdy[k] * dtdx[k] + dvgdy[k] * dtdy[k]);
    }
    (q1, q2)
}

/// Q-vector convergence: -2∇·Q = -2(∂Q1/∂x + ∂Q2/∂y).
pub fn q_vector_convergence(
    q1: &[f64], q2: &[f64],
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    let dq1dx = gradient_x(q1, nx, ny, dx);
    let dq2dy = gradient_y(q2, nx, ny, dy);
    dq1dx
        .par_iter()
        .zip(dq2dy.par_iter())
        .map(|(a, b)| -2.0 * (a + b))
        .collect()
}

// ─────────────────────────────────────────────
// Wind utilities
// ─────────────────────────────────────────────

/// Wind speed: √(u² + v²).
pub fn wind_speed(u: &[f64], v: &[f64]) -> Vec<f64> {
    assert_eq!(u.len(), v.len());
    u.par_iter()
        .zip(v.par_iter())
        .map(|(ui, vi)| (ui * ui + vi * vi).sqrt())
        .collect()
}

/// Meteorological wind direction (degrees, 0 = from north, 90 = from east).
/// Returns the direction the wind is coming FROM.
pub fn wind_direction(u: &[f64], v: &[f64]) -> Vec<f64> {
    assert_eq!(u.len(), v.len());
    u.par_iter()
        .zip(v.par_iter())
        .map(|(ui, vi)| {
            let spd = (ui * ui + vi * vi).sqrt();
            if spd < 1e-10 {
                0.0
            } else {
                let dir = (ui.atan2(*vi) * 180.0 / PI) + 180.0;
                dir % 360.0
            }
        })
        .collect()
}

/// Convert wind speed and meteorological direction to (u, v) components.
/// Direction is in degrees (0 = from north).
pub fn wind_components(speed: &[f64], direction: &[f64]) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(speed.len(), direction.len());
    let (u, v): (Vec<f64>, Vec<f64>) = speed
        .par_iter()
        .zip(direction.par_iter())
        .map(|(s, d)| {
            let rad = d * PI / 180.0;
            (-s * rad.sin(), -s * rad.cos())
        })
        .unzip();
    (u, v)
}

/// Geostrophic wind from geopotential height field (m).
///
/// u_g = -(g/f) · ∂Z/∂y
/// v_g =  (g/f) · ∂Z/∂x
///
/// `lats` is a flattened array of latitudes (degrees) at each grid point.
/// Returns (u_geo, v_geo).
pub fn geostrophic_wind(
    height: &[f64], lats: &[f64],
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> (Vec<f64>, Vec<f64>) {
    let dzdx = gradient_x(height, nx, ny, dx);
    let dzdy = gradient_y(height, nx, ny, dy);
    let n = nx * ny;
    let mut u_geo = vec![0.0; n];
    let mut v_geo = vec![0.0; n];
    for k in 0..n {
        let f = coriolis_parameter(lats[k]);
        if f.abs() < 1e-10 {
            // Near equator, geostrophic balance breaks down
            u_geo[k] = 0.0;
            v_geo[k] = 0.0;
        } else {
            let gf = G / f;
            u_geo[k] = -gf * dzdy[k];
            v_geo[k] = gf * dzdx[k];
        }
    }
    (u_geo, v_geo)
}

// ─────────────────────────────────────────────
// Ageostrophic wind
// ─────────────────────────────────────────────

/// Ageostrophic wind: (u - u_geo, v - v_geo).
pub fn ageostrophic_wind(
    u: &[f64], v: &[f64], u_geo: &[f64], v_geo: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(u.len(), v.len());
    assert_eq!(u.len(), u_geo.len());
    assert_eq!(u.len(), v_geo.len());
    let ua: Vec<f64> = u.par_iter().zip(u_geo.par_iter()).map(|(a, b)| a - b).collect();
    let va: Vec<f64> = v.par_iter().zip(v_geo.par_iter()).map(|(a, b)| a - b).collect();
    (ua, va)
}

// ─────────────────────────────────────────────
// Curvature and shear vorticity
// ─────────────────────────────────────────────

/// Curvature vorticity — the component of vorticity arising from
/// curvature of the streamlines.
///
/// For a 2D wind field (u, v):
///   V = √(u² + v²)
///   ψ = atan2(v, u)   (wind direction angle)
///   ζ_c = V · (∂ψ/∂s)  where s is along-stream
///       = V · [ (∂ψ/∂x)(u/V) + (∂ψ/∂y)(v/V) ]
///       = u·(∂ψ/∂x) + v·(∂ψ/∂y)
///
/// ψ derivatives are computed via the chain rule:
///   ∂ψ/∂x = (u·∂v/∂x - v·∂u/∂x) / V²
///   ∂ψ/∂y = (u·∂v/∂y - v·∂u/∂y) / V²
pub fn curvature_vorticity(
    u: &[f64], v: &[f64], nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(u.len(), n);
    assert_eq!(v.len(), n);

    let dudx = gradient_x(u, nx, ny, dx);
    let dudy = gradient_y(u, nx, ny, dy);
    let dvdx = gradient_x(v, nx, ny, dx);
    let dvdy = gradient_y(v, nx, ny, dy);

    let mut out = vec![0.0; n];
    for k in 0..n {
        let spd2 = u[k] * u[k] + v[k] * v[k];
        if spd2 < 1e-20 {
            out[k] = 0.0;
        } else {
            // ∂ψ/∂x = (u dvdx - v dudx) / V²
            let dpsidx = (u[k] * dvdx[k] - v[k] * dudx[k]) / spd2;
            // ∂ψ/∂y = (u dvdy - v dudy) / V²
            let dpsidy = (u[k] * dvdy[k] - v[k] * dudy[k]) / spd2;
            // ζ_c = u * ∂ψ/∂x + v * ∂ψ/∂y
            out[k] = u[k] * dpsidx + v[k] * dpsidy;
        }
    }
    out
}

/// Shear vorticity — the component of vorticity arising from speed
/// shear normal to the flow.
///
/// ζ_s = ζ - ζ_c  (total relative vorticity minus curvature vorticity)
pub fn shear_vorticity(
    u: &[f64], v: &[f64], nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    let total = vorticity(u, v, nx, ny, dx, dy);
    let curv = curvature_vorticity(u, v, nx, ny, dx, dy);
    total.par_iter().zip(curv.par_iter()).map(|(t, c)| t - c).collect()
}

// ─────────────────────────────────────────────
// Advanced dynamics
// ─────────────────────────────────────────────

/// Inertial-advective wind: the component of ageostrophic wind due to
/// inertial advection of the geostrophic wind.
///
/// u_ia = u · ∂(u-u_g)/∂x + v · ∂(u-u_g)/∂y  ... actually the standard
/// formulation is:
///   V_ia = -(1/f) * (V · ∇) V_g  (where V is the total wind, V_g geostrophic)
///
/// Since this requires the Coriolis parameter, and the specification doesn't
/// include lats, we compute the simpler kinematic form:
///   u_ia = u·∂u_g/∂x + v·∂u_g/∂y
///   v_ia = u·∂v_g/∂x + v·∂v_g/∂y
///
/// This is the advection of the geostrophic wind by the total wind.
pub fn inertial_advective_wind(
    u: &[f64], v: &[f64], u_geo: &[f64], v_geo: &[f64],
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = nx * ny;
    assert_eq!(u.len(), n);
    assert_eq!(v.len(), n);
    assert_eq!(u_geo.len(), n);
    assert_eq!(v_geo.len(), n);

    let dugdx = gradient_x(u_geo, nx, ny, dx);
    let dugdy = gradient_y(u_geo, nx, ny, dy);
    let dvgdx = gradient_x(v_geo, nx, ny, dx);
    let dvgdy = gradient_y(v_geo, nx, ny, dy);

    let mut u_ia = vec![0.0; n];
    let mut v_ia = vec![0.0; n];
    for k in 0..n {
        u_ia[k] = u[k] * dugdx[k] + v[k] * dugdy[k];
        v_ia[k] = u[k] * dvgdx[k] + v[k] * dvgdy[k];
    }
    (u_ia, v_ia)
}

/// Absolute momentum: M = u - f·y
///
/// `u`: zonal wind component (flattened grid).
/// `lats`: latitude in degrees at each grid point.
/// `y_distances`: distance from some reference latitude in meters.
pub fn absolute_momentum(
    u: &[f64], lats: &[f64], y_distances: &[f64],
) -> Vec<f64> {
    assert_eq!(u.len(), lats.len());
    assert_eq!(u.len(), y_distances.len());
    u.par_iter()
        .zip(lats.par_iter().zip(y_distances.par_iter()))
        .map(|(&ui, (&lat, &y))| {
            let f = coriolis_parameter(lat);
            ui - f * y
        })
        .collect()
}

/// Kinematic flux: element-wise product of a velocity component and a scalar.
///
/// Commonly used for turbulent flux calculations (e.g., u'θ', v'q').
pub fn kinematic_flux(v_component: &[f64], scalar: &[f64]) -> Vec<f64> {
    assert_eq!(v_component.len(), scalar.len());
    v_component
        .par_iter()
        .zip(scalar.par_iter())
        .map(|(v, s)| v * s)
        .collect()
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple 4x3 grid (nx=4, ny=3) with f(i,j) = i + j.
    /// Row-major: [0,1,2,3, 1,2,3,4, 2,3,4,5]
    fn make_linear_grid() -> (Vec<f64>, usize, usize) {
        let nx = 4;
        let ny = 3;
        let mut vals = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                vals[j * nx + i] = (i + j) as f64;
            }
        }
        (vals, nx, ny)
    }

    #[test]
    fn test_gradient_x_linear() {
        let (vals, nx, ny) = make_linear_grid();
        let dx = 1.0;
        let gx = gradient_x(&vals, nx, ny, dx);
        // For f = i + j, ∂f/∂x = 1 everywhere (centered and boundary)
        for j in 0..ny {
            for i in 0..nx {
                let val = gx[j * nx + i];
                assert!(
                    (val - 1.0).abs() < 1e-10,
                    "gradient_x at ({},{}) = {}, expected 1.0",
                    i, j, val
                );
            }
        }
    }

    #[test]
    fn test_gradient_y_linear() {
        let (vals, nx, ny) = make_linear_grid();
        let dy = 1.0;
        let gy = gradient_y(&vals, nx, ny, dy);
        // For f = i + j, ∂f/∂y = 1 everywhere
        for j in 0..ny {
            for i in 0..nx {
                let val = gy[j * nx + i];
                assert!(
                    (val - 1.0).abs() < 1e-10,
                    "gradient_y at ({},{}) = {}, expected 1.0",
                    i, j, val
                );
            }
        }
    }

    #[test]
    fn test_gradient_x_with_spacing() {
        // f = 2*i, dx = 0.5 => ∂f/∂x = 2/0.5 = 4... no
        // Actually f = 2*i means spacing in index is 2.
        // With dx = 2.0: centered diff = (2*(i+1) - 2*(i-1)) / (2*2) = 4/4 = 1.
        let nx = 5;
        let ny = 1;
        let vals: Vec<f64> = (0..5).map(|i| 2.0 * i as f64).collect();
        let dx = 2.0;
        let gx = gradient_x(&vals, nx, ny, dx);
        // Interior: (2*(i+1) - 2*(i-1)) / (2*2) = 4/4 = 1
        for i in 1..4 {
            assert!((gx[i] - 1.0).abs() < 1e-10);
        }
        // Boundaries: forward/backward => (2*1 - 2*0)/2 = 1, same
        assert!((gx[0] - 1.0).abs() < 1e-10);
        assert!((gx[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_divergence_uniform() {
        // Uniform wind: u=5, v=3. Divergence = 0.
        let nx = 5;
        let ny = 4;
        let n = nx * ny;
        let u = vec![5.0; n];
        let v = vec![3.0; n];
        let div = divergence(&u, &v, nx, ny, 1000.0, 1000.0);
        for val in &div {
            assert!(val.abs() < 1e-10, "divergence of uniform wind should be 0");
        }
    }

    #[test]
    fn test_divergence_expanding() {
        // u = x, v = y => div = 2
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let dx = 1.0;
        let dy = 1.0;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = i as f64;
                v[j * nx + i] = j as f64;
            }
        }
        let div = divergence(&u, &v, nx, ny, dx, dy);
        // Interior points should have divergence = 2
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    (div[j * nx + i] - 2.0).abs() < 1e-10,
                    "divergence at interior ({},{}) = {}, expected 2.0",
                    i, j, div[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_vorticity_solid_rotation() {
        // u = -y, v = x => vorticity = ∂v/∂x - ∂u/∂y = 1 - (-1) = 2
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
        // Interior points
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
    fn test_coriolis_parameter() {
        // At 45°N: f = 2 * 7.2921159e-5 * sin(45°) ≈ 1.0313e-4
        let f = coriolis_parameter(45.0);
        let expected = 2.0 * 7.2921159e-5 * (45.0_f64 * PI / 180.0).sin();
        assert!((f - expected).abs() < 1e-12);

        // At equator: f = 0
        let f_eq = coriolis_parameter(0.0);
        assert!(f_eq.abs() < 1e-15);

        // At pole: f = 2Ω
        let f_pole = coriolis_parameter(90.0);
        assert!((f_pole - 2.0 * OMEGA).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn test_absolute_vorticity_lat_mismatch_panics() {
        let nx = 2;
        let ny = 2;
        let u = vec![0.0; nx * ny];
        let v = vec![0.0; nx * ny];
        let lats = vec![35.0; 3];
        let _ = absolute_vorticity(&u, &v, &lats, nx, ny, 1.0, 1.0);
    }

    #[test]
    fn test_advection_uniform_scalar() {
        // Uniform scalar => gradient = 0 => advection = 0
        let nx = 5;
        let ny = 4;
        let n = nx * ny;
        let scalar = vec![10.0; n];
        let u = vec![5.0; n];
        let v = vec![3.0; n];
        let adv = advection(&scalar, &u, &v, nx, ny, 1000.0, 1000.0);
        for val in &adv {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_advection_linear_scalar() {
        // scalar = x, u = 1, v = 0, dx = 1 => advection = -1 * 1 = -1
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
        // Interior points: ∂s/∂x = 1, advection = -1*1 - 0*0 = -1
        for j in 0..ny {
            for i in 1..nx - 1 {
                assert!(
                    (adv[j * nx + i] - (-1.0)).abs() < 1e-10,
                    "advection at ({},{}) = {}, expected -1.0",
                    i, j, adv[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_wind_speed() {
        let u = vec![3.0, 0.0, -4.0];
        let v = vec![4.0, 5.0, 3.0];
        let spd = wind_speed(&u, &v);
        assert!((spd[0] - 5.0).abs() < 1e-10);
        assert!((spd[1] - 5.0).abs() < 1e-10);
        assert!((spd[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_wind_direction() {
        // Pure southerly wind (from south, blowing north): u=0, v>0
        // Meteorological direction = 180° (from south)
        let u = vec![0.0];
        let v = vec![10.0];
        let dir = wind_direction(&u, &v);
        assert!((dir[0] - 180.0).abs() < 1e-10, "southerly wind dir = {}", dir[0]);

        // Pure westerly wind (from west, blowing east): u>0, v=0
        // Meteorological direction = 270° (from west)
        let u = vec![10.0];
        let v = vec![0.0];
        let dir = wind_direction(&u, &v);
        assert!((dir[0] - 270.0).abs() < 1e-10, "westerly wind dir = {}", dir[0]);
    }

    #[test]
    fn test_wind_components_roundtrip() {
        let speed = vec![10.0, 20.0, 15.0];
        let direction = vec![180.0, 270.0, 45.0];
        let (u, v) = wind_components(&speed, &direction);

        // Roundtrip: compute speed and direction back
        let spd_back = wind_speed(&u, &v);
        let dir_back = wind_direction(&u, &v);

        for k in 0..3 {
            assert!(
                (spd_back[k] - speed[k]).abs() < 1e-10,
                "speed roundtrip failed at {}: {} vs {}",
                k, spd_back[k], speed[k]
            );
            assert!(
                (dir_back[k] - direction[k]).abs() < 1e-6,
                "direction roundtrip failed at {}: {} vs {}",
                k, dir_back[k], direction[k]
            );
        }
    }

    #[test]
    fn test_laplacian_quadratic() {
        // f = x² + y² => ∇²f = 2 + 2 = 4
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut vals = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                vals[j * nx + i] = (i * i + j * j) as f64;
            }
        }
        let lap = laplacian(&vals, nx, ny, 1.0, 1.0);
        // All points should have laplacian = 4 (for quadratic,
        // the second-order finite difference is exact)
        for j in 0..ny {
            for i in 0..nx {
                assert!(
                    (lap[j * nx + i] - 4.0).abs() < 1e-10,
                    "laplacian at ({},{}) = {}, expected 4.0",
                    i, j, lap[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_curvature_vorticity_solid_rotation() {
        // Solid-body rotation: u = -y, v = x
        // Vorticity = 2 everywhere. For solid rotation, curvature vorticity = 1
        // and shear vorticity = 1 (each contributes half).
        // Actually, for solid-body rotation: streamlines are circles,
        // curvature vorticity ζ_c = V/R where V = R·ω, R = distance.
        // For u = -y, v = x at (x,y), V = sqrt(x²+y²), R = V/ω = V/1 = V
        // so ζ_c = V/(V/1) = 1.  ζ_s = ζ - ζ_c = 2 - 1 = 1.
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        // Center the rotation at (3,3)
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 - 3.0;
                let y = j as f64 - 3.0;
                u[j * nx + i] = -y;
                v[j * nx + i] = x;
            }
        }
        let curv = curvature_vorticity(&u, &v, nx, ny, 1.0, 1.0);
        let shear = shear_vorticity(&u, &v, nx, ny, 1.0, 1.0);
        // Check an interior point away from center (center has V=0, skip it)
        // At (4,3): x=1, y=0, u=0, v=1
        let k = 3 * nx + 4; // j=3, i=4
        assert!(
            (curv[k] - 1.0).abs() < 0.2,
            "curvature vorticity at (4,3) = {}, expected ~1.0",
            curv[k]
        );
        assert!(
            (shear[k] - 1.0).abs() < 0.2,
            "shear vorticity at (4,3) = {}, expected ~1.0",
            shear[k]
        );
        // Their sum should be ~2.0 (total vorticity)
        assert!(
            (curv[k] + shear[k] - 2.0).abs() < 1e-10,
            "curv + shear = {}, expected 2.0",
            curv[k] + shear[k]
        );
    }

    #[test]
    fn test_total_deformation() {
        // Pure stretching: u = x, v = -y => stretching = 2, shearing = 0
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
        let td = total_deformation(&u, &v, nx, ny, 1.0, 1.0);
        // Interior: stretching = 1 - (-1) = 2, shearing = 0, total = 2
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    (td[j * nx + i] - 2.0).abs() < 1e-10,
                    "total_deformation at ({},{}) = {}, expected 2.0",
                    i, j, td[j * nx + i]
                );
            }
        }
    }

    // =========================================================================
    // Vorticity on solid body rotation (comprehensive)
    // =========================================================================

    #[test]
    fn test_vorticity_solid_rotation_scaled() {
        // u = -omega*y, v = omega*x => vort = 2*omega
        // Use omega = 3.5 and dx=dy=100m
        let omega = 3.5;
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let dx = 100.0;
        let dy = 100.0;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                u[j * nx + i] = -omega * y;
                v[j * nx + i] = omega * x;
            }
        }
        let vort = vorticity(&u, &v, nx, ny, dx, dy);
        // All interior points should have vorticity = 2*omega
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    (vort[j * nx + i] - 2.0 * omega).abs() < 1e-10,
                    "vorticity at ({},{}) = {}, expected {}",
                    i, j, vort[j * nx + i], 2.0 * omega
                );
            }
        }
    }

    #[test]
    fn test_vorticity_irrotational_field() {
        // u = x, v = y (pure divergence, no rotation) => vort = 0
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
        let vort = vorticity(&u, &v, nx, ny, 1.0, 1.0);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    vort[j * nx + i].abs() < 1e-10,
                    "vorticity of irrotational field at ({},{}) = {}, expected 0",
                    i, j, vort[j * nx + i]
                );
            }
        }
    }

    // =========================================================================
    // Divergence on expanding/contracting fields
    // =========================================================================

    #[test]
    fn test_divergence_contracting() {
        // u = -x, v = -y => div = -2
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = -(i as f64);
                v[j * nx + i] = -(j as f64);
            }
        }
        let div = divergence(&u, &v, nx, ny, 1.0, 1.0);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    (div[j * nx + i] - (-2.0)).abs() < 1e-10,
                    "divergence at ({},{}) = {}, expected -2.0",
                    i, j, div[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_divergence_scaled_expanding() {
        // u = alpha*x, v = beta*y => div = alpha + beta
        let alpha = 2.5;
        let beta = 1.3;
        let nx = 7;
        let ny = 7;
        let n = nx * ny;
        let dx = 50.0;
        let dy = 50.0;
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                u[j * nx + i] = alpha * x;
                v[j * nx + i] = beta * y;
            }
        }
        let div = divergence(&u, &v, nx, ny, dx, dy);
        let expected = alpha + beta;
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    (div[j * nx + i] - expected).abs() < 1e-10,
                    "divergence at ({},{}) = {}, expected {}",
                    i, j, div[j * nx + i], expected
                );
            }
        }
    }

    // =========================================================================
    // Curl of gradient = 0
    // =========================================================================

    #[test]
    fn test_curl_of_gradient_is_zero() {
        // For any smooth scalar field phi, curl(grad(phi)) = 0.
        // phi = x^2 + 3*x*y + y^2
        // grad_x = 2x + 3y, grad_y = 3x + 2y
        // curl = d(grad_y)/dx - d(grad_x)/dy = 3 - 3 = 0
        let nx = 9;
        let ny = 9;
        let n = nx * ny;
        let dx = 1.0;
        let dy = 1.0;

        // Compute grad(phi) analytically
        let mut grad_x_field = vec![0.0; n];
        let mut grad_y_field = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64;
                let y = j as f64;
                grad_x_field[j * nx + i] = 2.0 * x + 3.0 * y;
                grad_y_field[j * nx + i] = 3.0 * x + 2.0 * y;
            }
        }

        // curl = d(grad_y)/dx - d(grad_x)/dy
        let vort = vorticity(&grad_x_field, &grad_y_field, nx, ny, dx, dy);
        // Note: vorticity computes dv/dx - du/dy, where we pass u=grad_x, v=grad_y
        // So it computes d(grad_y)/dx - d(grad_x)/dy = curl(grad(phi))
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    vort[j * nx + i].abs() < 1e-10,
                    "curl(grad(phi)) at ({},{}) = {}, expected 0",
                    i, j, vort[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_curl_of_gradient_cubic() {
        // phi = x^3 + y^3 + x*y^2
        // grad_x = 3x^2 + y^2, grad_y = 3y^2 + 2xy
        // d(grad_y)/dx = 2y, d(grad_x)/dy = 2y => curl = 0
        let nx = 11;
        let ny = 11;
        let n = nx * ny;
        let dx = 0.5;
        let dy = 0.5;

        let mut phi = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                phi[j * nx + i] = x * x * x + y * y * y + x * y * y;
            }
        }

        // Compute gradient numerically
        let grad_x_num = gradient_x(&phi, nx, ny, dx);
        let grad_y_num = gradient_y(&phi, nx, ny, dy);

        // Curl of numerical gradient
        let curl = vorticity(&grad_x_num, &grad_y_num, nx, ny, dx, dy);

        // Interior points (away from boundaries where finite differences are less accurate)
        for j in 2..ny - 2 {
            for i in 2..nx - 2 {
                assert!(
                    curl[j * nx + i].abs() < 1e-6,
                    "curl(grad(phi)) at ({},{}) = {}, expected ~0",
                    i, j, curl[j * nx + i]
                );
            }
        }
    }
}
