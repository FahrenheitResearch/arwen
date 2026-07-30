/// Grid mathematics: generalized derivatives, geospatial gradient/laplacian,
/// and lat/lon grid utilities.
///
/// All grid arrays are flattened row-major: index = j * nx + i
/// where j is the y-index (row) and i is the x-index (column).


/// Earth's mean radius in meters.
const EARTH_RADIUS: f64 = 6_371_000.0;

// ─────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────

#[inline(always)]
fn idx(j: usize, i: usize, nx: usize) -> usize {
    j * nx + i
}

// ─────────────────────────────────────────────
// Generalized first / second derivative
// ─────────────────────────────────────────────

/// Generalized first derivative along axis 0 (x) or axis 1 (y).
///
/// Uses centered differences in the interior, forward/backward at boundaries.
/// `axis_spacing` is the uniform grid spacing along the chosen axis.
/// `axis`: 0 = x (columns), 1 = y (rows).
pub fn first_derivative(
    values: &[f64], axis_spacing: f64, axis: usize,
    nx: usize, ny: usize,
) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny);
    assert!(axis <= 1, "axis must be 0 (x) or 1 (y)");

    let mut out = vec![0.0; nx * ny];
    let inv_2h = 1.0 / (2.0 * axis_spacing);
    let inv_h = 1.0 / axis_spacing;

    if axis == 0 {
        // derivative along x
        for j in 0..ny {
            for i in 0..nx {
                let d = if nx < 2 {
                    0.0
                } else if nx == 2 {
                    // Only 2 points: first-order is all we can do
                    (values[idx(j, 1, nx)] - values[idx(j, 0, nx)]) * inv_h
                } else if i == 0 {
                    // 2nd-order forward: (-3f[0] + 4f[1] - f[2]) / (2h)
                    (-3.0 * values[idx(j, 0, nx)] + 4.0 * values[idx(j, 1, nx)]
                        - values[idx(j, 2, nx)]) * inv_2h
                } else if i == nx - 1 {
                    // 2nd-order backward: (3f[n] - 4f[n-1] + f[n-2]) / (2h)
                    (3.0 * values[idx(j, nx - 1, nx)] - 4.0 * values[idx(j, nx - 2, nx)]
                        + values[idx(j, nx - 3, nx)]) * inv_2h
                } else {
                    (values[idx(j, i + 1, nx)] - values[idx(j, i - 1, nx)]) * inv_2h
                };
                out[idx(j, i, nx)] = d;
            }
        }
    } else {
        // derivative along y
        for j in 0..ny {
            for i in 0..nx {
                let d = if ny < 2 {
                    0.0
                } else if ny == 2 {
                    (values[idx(1, i, nx)] - values[idx(0, i, nx)]) * inv_h
                } else if j == 0 {
                    // 2nd-order forward: (-3f[0] + 4f[1] - f[2]) / (2h)
                    (-3.0 * values[idx(0, i, nx)] + 4.0 * values[idx(1, i, nx)]
                        - values[idx(2, i, nx)]) * inv_2h
                } else if j == ny - 1 {
                    // 2nd-order backward: (3f[n] - 4f[n-1] + f[n-2]) / (2h)
                    (3.0 * values[idx(ny - 1, i, nx)] - 4.0 * values[idx(ny - 2, i, nx)]
                        + values[idx(ny - 3, i, nx)]) * inv_2h
                } else {
                    (values[idx(j + 1, i, nx)] - values[idx(j - 1, i, nx)]) * inv_2h
                };
                out[idx(j, i, nx)] = d;
            }
        }
    }
    out
}

/// Generalized second derivative along axis 0 (x) or axis 1 (y).
///
/// Uses centered second-order finite difference in the interior,
/// forward/backward at boundaries.
pub fn second_derivative(
    values: &[f64], axis_spacing: f64, axis: usize,
    nx: usize, ny: usize,
) -> Vec<f64> {
    assert_eq!(values.len(), nx * ny);
    assert!(axis <= 1, "axis must be 0 (x) or 1 (y)");

    let mut out = vec![0.0; nx * ny];
    let inv_h2 = 1.0 / (axis_spacing * axis_spacing);

    if axis == 0 {
        for j in 0..ny {
            for i in 0..nx {
                let d2 = if nx < 3 {
                    0.0
                } else if i == 0 {
                    (values[idx(j, 2, nx)] - 2.0 * values[idx(j, 1, nx)] + values[idx(j, 0, nx)]) * inv_h2
                } else if i == nx - 1 {
                    (values[idx(j, nx - 1, nx)] - 2.0 * values[idx(j, nx - 2, nx)] + values[idx(j, nx - 3, nx)]) * inv_h2
                } else {
                    (values[idx(j, i + 1, nx)] - 2.0 * values[idx(j, i, nx)] + values[idx(j, i - 1, nx)]) * inv_h2
                };
                out[idx(j, i, nx)] = d2;
            }
        }
    } else {
        for j in 0..ny {
            for i in 0..nx {
                let d2 = if ny < 3 {
                    0.0
                } else if j == 0 {
                    (values[idx(2, i, nx)] - 2.0 * values[idx(1, i, nx)] + values[idx(0, i, nx)]) * inv_h2
                } else if j == ny - 1 {
                    (values[idx(ny - 1, i, nx)] - 2.0 * values[idx(ny - 2, i, nx)] + values[idx(ny - 3, i, nx)]) * inv_h2
                } else {
                    (values[idx(j + 1, i, nx)] - 2.0 * values[idx(j, i, nx)] + values[idx(j - 1, i, nx)]) * inv_h2
                };
                out[idx(j, i, nx)] = d2;
            }
        }
    }
    out
}

// ─────────────────────────────────────────────
// Haversine-based grid deltas
// ─────────────────────────────────────────────

/// Haversine distance between two points in meters.
fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let lat1r = lat1.to_radians();
    let lat2r = lat2.to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1r.cos() * lat2r.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS * a.sqrt().asin()
}

/// Compute physical grid spacings (dx, dy) in meters from lat/lon arrays.
///
/// `lats` and `lons` are flattened row-major arrays of length `nx * ny`.
/// Returns `(dx, dy)` where each is a flattened row-major array of the same
/// size. `dx[k]` is the local east-west spacing and `dy[k]` is the local
/// north-south spacing (both in meters).
///
/// At boundaries, one-sided differences are used.
pub fn lat_lon_grid_deltas(
    lats: &[f64], lons: &[f64], nx: usize, ny: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = nx * ny;
    assert_eq!(lats.len(), n);
    assert_eq!(lons.len(), n);

    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; n];

    // dx: spacing along x (columns) at each grid point
    for j in 0..ny {
        for i in 0..nx {
            let d = if nx < 2 {
                0.0
            } else if i == 0 {
                haversine(
                    lats[idx(j, 0, nx)], lons[idx(j, 0, nx)],
                    lats[idx(j, 1, nx)], lons[idx(j, 1, nx)],
                )
            } else if i == nx - 1 {
                haversine(
                    lats[idx(j, nx - 2, nx)], lons[idx(j, nx - 2, nx)],
                    lats[idx(j, nx - 1, nx)], lons[idx(j, nx - 1, nx)],
                )
            } else {
                haversine(
                    lats[idx(j, i - 1, nx)], lons[idx(j, i - 1, nx)],
                    lats[idx(j, i + 1, nx)], lons[idx(j, i + 1, nx)],
                ) / 2.0
            };
            dx[idx(j, i, nx)] = d;
        }
    }

    // dy: spacing along y (rows) at each grid point
    for j in 0..ny {
        for i in 0..nx {
            let d = if ny < 2 {
                0.0
            } else if j == 0 {
                haversine(
                    lats[idx(0, i, nx)], lons[idx(0, i, nx)],
                    lats[idx(1, i, nx)], lons[idx(1, i, nx)],
                )
            } else if j == ny - 1 {
                haversine(
                    lats[idx(ny - 2, i, nx)], lons[idx(ny - 2, i, nx)],
                    lats[idx(ny - 1, i, nx)], lons[idx(ny - 1, i, nx)],
                )
            } else {
                haversine(
                    lats[idx(j - 1, i, nx)], lons[idx(j - 1, i, nx)],
                    lats[idx(j + 1, i, nx)], lons[idx(j + 1, i, nx)],
                ) / 2.0
            };
            dy[idx(j, i, nx)] = d;
        }
    }

    (dx, dy)
}

// ─────────────────────────────────────────────
// Geospatial gradient
// ─────────────────────────────────────────────

/// Gradient of a scalar field on a lat/lon grid, accounting for varying
/// grid spacing.
///
/// Returns `(df_dx, df_dy)` in physical units (per meter). Uses the
/// haversine-derived local grid spacings for each point.
pub fn geospatial_gradient(
    values: &[f64], lats: &[f64], lons: &[f64],
    nx: usize, ny: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = nx * ny;
    assert_eq!(values.len(), n);
    let (local_dx, local_dy) = lat_lon_grid_deltas(lats, lons, nx, ny);

    let mut dfdx = vec![0.0; n];
    let mut dfdy = vec![0.0; n];

    // df/dx
    for j in 0..ny {
        for i in 0..nx {
            let k = idx(j, i, nx);
            let dx_m = local_dx[k];
            if dx_m < 1e-10 || nx < 2 {
                dfdx[k] = 0.0;
            } else if nx == 2 {
                dfdx[k] = (values[idx(j, 1, nx)] - values[idx(j, 0, nx)]) / dx_m;
            } else if i == 0 {
                // 2nd-order forward: (-3f[0] + 4f[1] - f[2]) / (2dx)
                dfdx[k] = (-3.0 * values[idx(j, 0, nx)] + 4.0 * values[idx(j, 1, nx)]
                    - values[idx(j, 2, nx)]) / (2.0 * dx_m);
            } else if i == nx - 1 {
                // 2nd-order backward: (3f[n] - 4f[n-1] + f[n-2]) / (2dx)
                dfdx[k] = (3.0 * values[idx(j, nx - 1, nx)] - 4.0 * values[idx(j, nx - 2, nx)]
                    + values[idx(j, nx - 3, nx)]) / (2.0 * dx_m);
            } else {
                // centered: dx_m is the average spacing, difference spans 2*dx_m
                dfdx[k] = (values[idx(j, i + 1, nx)] - values[idx(j, i - 1, nx)]) / (2.0 * dx_m);
            }
        }
    }

    // df/dy
    for j in 0..ny {
        for i in 0..nx {
            let k = idx(j, i, nx);
            let dy_m = local_dy[k];
            if dy_m < 1e-10 || ny < 2 {
                dfdy[k] = 0.0;
            } else if ny == 2 {
                dfdy[k] = (values[idx(1, i, nx)] - values[idx(0, i, nx)]) / dy_m;
            } else if j == 0 {
                // 2nd-order forward: (-3f[0] + 4f[1] - f[2]) / (2dy)
                dfdy[k] = (-3.0 * values[idx(0, i, nx)] + 4.0 * values[idx(1, i, nx)]
                    - values[idx(2, i, nx)]) / (2.0 * dy_m);
            } else if j == ny - 1 {
                // 2nd-order backward: (3f[n] - 4f[n-1] + f[n-2]) / (2dy)
                dfdy[k] = (3.0 * values[idx(ny - 1, i, nx)] - 4.0 * values[idx(ny - 2, i, nx)]
                    + values[idx(ny - 3, i, nx)]) / (2.0 * dy_m);
            } else {
                dfdy[k] = (values[idx(j + 1, i, nx)] - values[idx(j - 1, i, nx)]) / (2.0 * dy_m);
            }
        }
    }

    (dfdx, dfdy)
}

// ─────────────────────────────────────────────
// Geospatial Laplacian
// ─────────────────────────────────────────────

/// Laplacian of a scalar field on a lat/lon grid (∂²f/∂x² + ∂²f/∂y²),
/// accounting for varying grid spacing via haversine distances.
pub fn geospatial_laplacian(
    values: &[f64], lats: &[f64], lons: &[f64],
    nx: usize, ny: usize,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(values.len(), n);
    let (local_dx, local_dy) = lat_lon_grid_deltas(lats, lons, nx, ny);

    let mut out = vec![0.0; n];

    for j in 0..ny {
        for i in 0..nx {
            let k = idx(j, i, nx);

            // ∂²f/∂x²
            let dx_m = local_dx[k];
            let d2x = if nx < 3 || dx_m < 1e-10 {
                0.0
            } else if i == 0 {
                (values[idx(j, 2, nx)] - 2.0 * values[idx(j, 1, nx)] + values[idx(j, 0, nx)])
                    / (dx_m * dx_m)
            } else if i == nx - 1 {
                (values[idx(j, nx - 1, nx)] - 2.0 * values[idx(j, nx - 2, nx)] + values[idx(j, nx - 3, nx)])
                    / (dx_m * dx_m)
            } else {
                (values[idx(j, i + 1, nx)] - 2.0 * values[idx(j, i, nx)] + values[idx(j, i - 1, nx)])
                    / (dx_m * dx_m)
            };

            // ∂²f/∂y²
            let dy_m = local_dy[k];
            let d2y = if ny < 3 || dy_m < 1e-10 {
                0.0
            } else if j == 0 {
                (values[idx(2, i, nx)] - 2.0 * values[idx(1, i, nx)] + values[idx(0, i, nx)])
                    / (dy_m * dy_m)
            } else if j == ny - 1 {
                (values[idx(ny - 1, i, nx)] - 2.0 * values[idx(ny - 2, i, nx)] + values[idx(ny - 3, i, nx)])
                    / (dy_m * dy_m)
            } else {
                (values[idx(j + 1, i, nx)] - 2.0 * values[idx(j, i, nx)] + values[idx(j - 1, i, nx)])
                    / (dy_m * dy_m)
            };

            out[k] = d2x + d2y;
        }
    }
    out
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_derivative_x() {
        // f = 2*i, dx = 1 => df/dx = 2
        let nx = 5;
        let ny = 3;
        let mut vals = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                vals[j * nx + i] = 2.0 * i as f64;
            }
        }
        let deriv = first_derivative(&vals, 1.0, 0, nx, ny);
        for j in 0..ny {
            for i in 0..nx {
                assert!(
                    (deriv[j * nx + i] - 2.0).abs() < 1e-10,
                    "first_derivative x at ({},{}) = {}, expected 2.0",
                    i, j, deriv[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_first_derivative_y() {
        // f = 3*j, dy = 1 => df/dy = 3
        let nx = 4;
        let ny = 5;
        let mut vals = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                vals[j * nx + i] = 3.0 * j as f64;
            }
        }
        let deriv = first_derivative(&vals, 1.0, 1, nx, ny);
        for j in 0..ny {
            for i in 0..nx {
                assert!(
                    (deriv[j * nx + i] - 3.0).abs() < 1e-10,
                    "first_derivative y at ({},{}) = {}, expected 3.0",
                    i, j, deriv[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_second_derivative_quadratic() {
        // f = i^2, dx = 1 => d2f/dx2 = 2 everywhere (exact for quadratic)
        let nx = 5;
        let ny = 3;
        let mut vals = vec![0.0; nx * ny];
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
    fn test_second_derivative_y_quadratic() {
        // f = j^2, dy = 1 => d2f/dy2 = 2
        let nx = 3;
        let ny = 5;
        let mut vals = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                vals[j * nx + i] = (j * j) as f64;
            }
        }
        let d2 = second_derivative(&vals, 1.0, 1, nx, ny);
        for j in 0..ny {
            for i in 0..nx {
                assert!(
                    (d2[j * nx + i] - 2.0).abs() < 1e-10,
                    "second_derivative y at ({},{}) = {}, expected 2.0",
                    i, j, d2[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_lat_lon_grid_deltas() {
        // A simple 3x3 grid at 1-degree spacing near 45N
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

        // At 45N, 1 degree of latitude ≈ 111.13 km
        // 1 degree of longitude ≈ 111.13 * cos(45°) ≈ 78.6 km
        let center_dy = dy[4]; // center point
        let center_dx = dx[4];

        assert!(
            (center_dy - 111_130.0).abs() < 500.0,
            "dy at center = {} m, expected ~111130",
            center_dy
        );
        assert!(
            (center_dx - 78_600.0).abs() < 1500.0,
            "dx at center = {} m, expected ~78600",
            center_dx
        );
    }

    #[test]
    fn test_geospatial_gradient() {
        // 3x3 grid, 1-degree spacing, scalar = latitude
        // df/dy should be ~1/(111km) = ~9e-6 per meter
        // df/dx should be ~0
        let nx = 3;
        let ny = 3;
        let mut lats = vec![0.0; 9];
        let mut lons = vec![0.0; 9];
        let mut vals = vec![0.0; 9];
        for j in 0..ny {
            for i in 0..nx {
                lats[j * nx + i] = 44.0 + j as f64;
                lons[j * nx + i] = -90.0 + i as f64;
                vals[j * nx + i] = 44.0 + j as f64; // scalar = latitude
            }
        }
        let (dfdx, dfdy) = geospatial_gradient(&vals, &lats, &lons, nx, ny);

        // Center point
        let center_dfdx = dfdx[4];
        let center_dfdy = dfdy[4];

        // df/dx should be ~0 since scalar doesn't vary in x
        assert!(
            center_dfdx.abs() < 1e-8,
            "dfdx at center = {}, expected ~0",
            center_dfdx
        );

        // df/dy: 1 degree change over ~111km => ~9e-6 /m
        let expected_dfdy = 1.0 / 111_130.0;
        assert!(
            (center_dfdy - expected_dfdy).abs() / expected_dfdy < 0.01,
            "dfdy at center = {}, expected ~{}",
            center_dfdy, expected_dfdy
        );
    }

    #[test]
    fn test_geospatial_laplacian_constant() {
        // Constant field => laplacian = 0
        let nx = 4;
        let ny = 4;
        let n = nx * ny;
        let mut lats = vec![0.0; n];
        let mut lons = vec![0.0; n];
        let vals = vec![42.0; n];
        for j in 0..ny {
            for i in 0..nx {
                lats[j * nx + i] = 40.0 + j as f64;
                lons[j * nx + i] = -90.0 + i as f64;
            }
        }
        let lap = geospatial_laplacian(&vals, &lats, &lons, nx, ny);
        for &v in &lap {
            assert!(v.abs() < 1e-15, "laplacian of constant = {}", v);
        }
    }

    // =========================================================================
    // First derivative on polynomial (x^2 -> 2x)
    // =========================================================================

    #[test]
    fn test_first_derivative_x_squared() {
        // f(i) = (i*dx)^2, df/dx = 2*i*dx
        // With dx=1: f = i^2, df/dx = 2*i (exact for centered differences on quadratic)
        let nx = 9;
        let ny = 1;
        let dx = 1.0;
        let mut vals = vec![0.0; nx];
        for i in 0..nx {
            vals[i] = (i as f64) * (i as f64);
        }
        let deriv = first_derivative(&vals, dx, 0, nx, ny);
        // Interior points: centered difference on x^2 is exact => 2*i
        for i in 1..nx - 1 {
            let expected = 2.0 * i as f64;
            assert!(
                (deriv[i] - expected).abs() < 1e-10,
                "d(x^2)/dx at i={}: got={}, expected={}",
                i, deriv[i], expected
            );
        }
    }

    #[test]
    fn test_first_derivative_x_squared_with_spacing() {
        // f(x) = x^2 where x = i*h, h = 0.25
        // df/dx = 2x = 2*i*h
        let nx = 11;
        let ny = 1;
        let h = 0.25;
        let mut vals = vec![0.0; nx];
        for i in 0..nx {
            let x = i as f64 * h;
            vals[i] = x * x;
        }
        let deriv = first_derivative(&vals, h, 0, nx, ny);
        for i in 1..nx - 1 {
            let x = i as f64 * h;
            let expected = 2.0 * x;
            assert!(
                (deriv[i] - expected).abs() < 1e-10,
                "d(x^2)/dx at x={}: got={}, expected={}",
                x, deriv[i], expected
            );
        }
    }

    #[test]
    fn test_first_derivative_cubic() {
        // f(x) = x^3, df/dx = 3x^2
        // Centered differences on cubic are exact (3-point stencil is exact for
        // polynomials up to degree 2, but for x^3 the centered diff gives
        // ((x+h)^3 - (x-h)^3)/(2h) = 3x^2 + h^2, so the error is h^2).
        // With h=0.1, error is 0.01. Use a tolerance that accounts for this.
        let nx = 21;
        let ny = 1;
        let h = 0.1;
        let mut vals = vec![0.0; nx];
        for i in 0..nx {
            let x = i as f64 * h;
            vals[i] = x * x * x;
        }
        let deriv = first_derivative(&vals, h, 0, nx, ny);
        // Interior centered-difference error for x^3 is exactly h^2 = 0.01
        for i in 1..nx - 1 {
            let x = i as f64 * h;
            let expected = 3.0 * x * x;
            assert!(
                (deriv[i] - expected).abs() < h * h + 1e-10,
                "d(x^3)/dx at x={:.1}: got={}, expected={}, tol={}",
                x, deriv[i], expected, h * h
            );
        }
    }

    // =========================================================================
    // Second derivative on polynomial (x^2 -> 2)
    // =========================================================================

    #[test]
    fn test_second_derivative_x_squared_exact() {
        // f = x^2, d2f/dx2 = 2 (exact for centered second difference)
        let nx = 11;
        let ny = 3;
        let h = 0.5;
        let mut vals = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * h;
                vals[j * nx + i] = x * x;
            }
        }
        let d2 = second_derivative(&vals, h, 0, nx, ny);
        // All points (boundary stencils are also exact for quadratics)
        for j in 0..ny {
            for i in 0..nx {
                assert!(
                    (d2[j * nx + i] - 2.0).abs() < 1e-10,
                    "d2(x^2)/dx2 at ({},{}): got={}, expected=2.0",
                    i, j, d2[j * nx + i]
                );
            }
        }
    }

    #[test]
    fn test_second_derivative_cubic() {
        // f = x^3, d2f/dx2 = 6x (exact for centered differences on cubic)
        let nx = 9;
        let ny = 1;
        let h = 1.0;
        let mut vals = vec![0.0; nx];
        for i in 0..nx {
            let x = i as f64 * h;
            vals[i] = x * x * x;
        }
        let d2 = second_derivative(&vals, h, 0, nx, ny);
        // Interior: centered second difference of x^3 is exactly 6x
        for i in 1..nx - 1 {
            let x = i as f64 * h;
            let expected = 6.0 * x;
            assert!(
                (d2[i] - expected).abs() < 1e-10,
                "d2(x^3)/dx2 at x={}: got={}, expected={}",
                x, d2[i], expected
            );
        }
    }

    // =========================================================================
    // Geospatial gradient on a tilted plane
    // =========================================================================

    #[test]
    fn test_geospatial_gradient_tilted_plane_lat() {
        // scalar = latitude => df/dx = 0, df/dy = 1/(dy_meters)
        // On a uniform lat/lon grid, dy ~ 111km per degree
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut lats = vec![0.0; n];
        let mut lons = vec![0.0; n];
        let mut vals = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                lats[j * nx + i] = 40.0 + j as f64;
                lons[j * nx + i] = -90.0 + i as f64;
                vals[j * nx + i] = 40.0 + j as f64; // scalar = latitude
            }
        }
        let (dfdx, dfdy) = geospatial_gradient(&vals, &lats, &lons, nx, ny);

        // Interior df/dx should be ~0
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    dfdx[j * nx + i].abs() < 1e-10,
                    "dfdx at ({},{}) = {}, expected ~0",
                    i, j, dfdx[j * nx + i]
                );
            }
        }

        // Interior df/dy should be ~1/111130 = ~9e-6 per meter
        let expected_dfdy = 1.0 / 111_130.0;
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let rel_err = (dfdy[j * nx + i] - expected_dfdy).abs() / expected_dfdy;
                assert!(
                    rel_err < 0.02,
                    "dfdy at ({},{}) = {:.3e}, expected ~{:.3e}, rel_err={}",
                    i, j, dfdy[j * nx + i], expected_dfdy, rel_err
                );
            }
        }
    }

    #[test]
    fn test_geospatial_gradient_tilted_plane_lon() {
        // scalar = longitude => df/dy = 0, df/dx = 1/(dx_meters)
        // At 45N, dx ~ 78.6km per degree of longitude
        let nx = 5;
        let ny = 5;
        let n = nx * ny;
        let mut lats = vec![0.0; n];
        let mut lons = vec![0.0; n];
        let mut vals = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                lats[j * nx + i] = 44.0 + j as f64;
                lons[j * nx + i] = -90.0 + i as f64;
                vals[j * nx + i] = -90.0 + i as f64; // scalar = longitude
            }
        }
        let (dfdx, dfdy) = geospatial_gradient(&vals, &lats, &lons, nx, ny);

        // Interior df/dy should be ~0
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    dfdy[j * nx + i].abs() < 1e-10,
                    "dfdy at ({},{}) = {}, expected ~0",
                    i, j, dfdy[j * nx + i]
                );
            }
        }

        // Interior df/dx should be 1/(local_dx in meters)
        // At ~45N, 1 degree longitude ~ 78.6 km
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                // dfdx should be positive and on order of 1/78600
                assert!(
                    dfdx[j * nx + i] > 0.0,
                    "dfdx at ({},{}) = {} should be positive",
                    i, j, dfdx[j * nx + i]
                );
                let approx_expected = 1.0 / 78_600.0;
                let rel_err = (dfdx[j * nx + i] - approx_expected).abs() / approx_expected;
                assert!(
                    rel_err < 0.05,
                    "dfdx at ({},{}) = {:.3e}, expected ~{:.3e}, rel_err={}",
                    i, j, dfdx[j * nx + i], approx_expected, rel_err
                );
            }
        }
    }
}
