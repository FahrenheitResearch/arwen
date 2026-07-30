/// Dynamics and kinematics calculations for 2D grids.
///
/// All grid arrays are flattened row-major: index = j * nx + i
/// where j is the y-index (row) and i is the x-index (column).
/// `dx` and `dy` are grid spacings in meters.
///
/// Vendored from wx-math crate for self-contained builds.
use std::f64::consts::PI;

/// Earth's angular velocity (rad/s).
const OMEGA: f64 = 7.2921e-5;

/// Specific gas constant for dry air (J/(kg*K)).
const RD: f64 = 287.058;

/// Gravitational acceleration (m/s^2).
const G: f64 = 9.80665;

// Helper: index into row-major grid
#[inline(always)]
fn idx(j: usize, i: usize, nx: usize) -> usize {
    j * nx + i
}

// Finite-difference derivatives

/// dF/dx using centered differences (forward/backward at boundaries).
pub fn gradient_x(values: &[f64], nx: usize, ny: usize, dx: f64) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny);
    let mut out = vec![0.0; nx * ny];
    let inv_2dx = 1.0 / (2.0 * dx);
    let inv_dx = 1.0 / dx;

    for j in 0..ny {
        for i in 0..nx {
            let grad = if nx < 2 {
                0.0
            } else if i == 0 {
                // forward difference
                (values[idx(j, 1, nx)] - values[idx(j, 0, nx)]) * inv_dx
            } else if i == nx - 1 {
                // backward difference
                (values[idx(j, nx - 1, nx)] - values[idx(j, nx - 2, nx)]) * inv_dx
            } else {
                // centered difference
                (values[idx(j, i + 1, nx)] - values[idx(j, i - 1, nx)]) * inv_2dx
            };
            out[idx(j, i, nx)] = grad;
        }
    }
    out
}

/// dF/dy using centered differences (forward/backward at boundaries).
pub fn gradient_y(values: &[f64], nx: usize, ny: usize, dy: f64) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny);
    let mut out = vec![0.0; nx * ny];
    let inv_2dy = 1.0 / (2.0 * dy);
    let inv_dy = 1.0 / dy;

    for j in 0..ny {
        for i in 0..nx {
            let grad = if ny < 2 {
                0.0
            } else if j == 0 {
                (values[idx(1, i, nx)] - values[idx(0, i, nx)]) * inv_dy
            } else if j == ny - 1 {
                (values[idx(ny - 1, i, nx)] - values[idx(ny - 2, i, nx)]) * inv_dy
            } else {
                (values[idx(j + 1, i, nx)] - values[idx(j - 1, i, nx)]) * inv_2dy
            };
            out[idx(j, i, nx)] = grad;
        }
    }
    out
}

/// Relative vorticity: dv/dx - du/dy.
pub fn vorticity(u: &[f64], v: &[f64], nx: usize, ny: usize, dx: f64, dy: f64) -> Vec<f64> {
    let dvdx = gradient_x(v, nx, ny, dx);
    let dudy = gradient_y(u, nx, ny, dy);
    dvdx.iter().zip(dudy.iter()).map(|(a, b)| a - b).collect()
}

/// Coriolis parameter: f = 2*Omega*sin(phi).
pub fn coriolis_parameter(lat_deg: f64) -> f64 {
    2.0 * OMEGA * (lat_deg * PI / 180.0).sin()
}

/// Absolute vorticity: relative vorticity + Coriolis parameter.
/// `lats` is a flattened array of latitudes (degrees) at each grid point.
pub fn absolute_vorticity(
    u: &[f64],
    v: &[f64],
    lats: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let rel = vorticity(u, v, nx, ny, dx, dy);
    assert_eq!(lats.len(), nx * ny);
    rel.iter()
        .zip(lats.iter())
        .map(|(zeta, lat)| zeta + coriolis_parameter(*lat))
        .collect()
}

/// Horizontal divergence: du/dx + dv/dy.
pub fn divergence(u: &[f64], v: &[f64], nx: usize, ny: usize, dx: f64, dy: f64) -> Vec<f64> {
    let dudx = gradient_x(u, nx, ny, dx);
    let dvdy = gradient_y(v, nx, ny, dy);
    dudx.iter().zip(dvdy.iter()).map(|(a, b)| a + b).collect()
}

/// Wind speed: sqrt(u^2 + v^2).
pub fn wind_speed(u: &[f64], v: &[f64]) -> Vec<f64> {
    assert_eq!(u.len(), v.len());
    u.iter()
        .zip(v.iter())
        .map(|(ui, vi)| (ui * ui + vi * vi).sqrt())
        .collect()
}

/// Meteorological wind direction (degrees, 0 = from north, 90 = from east).
/// Returns the direction the wind is coming FROM.
pub fn wind_direction(u: &[f64], v: &[f64]) -> Vec<f64> {
    assert_eq!(u.len(), v.len());
    u.iter()
        .zip(v.iter())
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
    let u: Vec<f64> = speed
        .iter()
        .zip(direction.iter())
        .map(|(s, d)| -s * (d * PI / 180.0).sin())
        .collect();
    let v: Vec<f64> = speed
        .iter()
        .zip(direction.iter())
        .map(|(s, d)| -s * (d * PI / 180.0).cos())
        .collect();
    (u, v)
}
