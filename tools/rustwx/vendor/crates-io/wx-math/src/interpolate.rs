//! Interpolation routines (Barnes, Cressman, IDW families).
//!
//! This module provides scattered-data interpolation functions that support
//! multiple weighting schemes via the `kind` parameter:
//!
//! * `kind = 0` -- standard inverse-distance weighting (1/d^2)
//! * `kind = 1` -- Barnes scheme (Gaussian weight controlled by `kappa` and `gamma`)
//! * `kind = 2` -- Cressman scheme (parabolic weight within `radius`)

/// Inverse distance weighted interpolation to arbitrary target points.
///
/// # Arguments
///
/// * `obs_x`  -- observation x-coordinates (e.g. longitude or projected easting)
/// * `obs_y`  -- observation y-coordinates (e.g. latitude or projected northing)
/// * `obs_values` -- observed values at each station
/// * `grid_x` -- target x-coordinates to interpolate onto
/// * `grid_y` -- target y-coordinates to interpolate onto
/// * `radius` -- search radius (same units as coordinates)
/// * `min_neighbors` -- minimum number of neighbors required; returns `NaN` if fewer
/// * `kind`   -- weighting scheme: 0 = IDW, 1 = Barnes, 2 = Cressman
/// * `kappa`  -- smoothing parameter for Barnes scheme
/// * `gamma`  -- convergence parameter for Barnes scheme
///
/// # Returns
///
/// A `Vec<f64>` of length `grid_x.len()` with the interpolated values.
pub fn inverse_distance_to_points(
    obs_x: &[f64],
    obs_y: &[f64],
    obs_values: &[f64],
    grid_x: &[f64],
    grid_y: &[f64],
    radius: f64,
    min_neighbors: usize,
    kind: u8,
    kappa: f64,
    gamma: f64,
) -> Vec<f64> {
    let n = obs_values.len();
    assert_eq!(obs_x.len(), n, "obs_x length must match obs_values");
    assert_eq!(obs_y.len(), n, "obs_y length must match obs_values");
    assert_eq!(grid_x.len(), grid_y.len(), "grid_x and grid_y must have the same length");

    let r2 = radius * radius;

    grid_x
        .iter()
        .zip(grid_y.iter())
        .map(|(&gx, &gy)| {
            let mut w_sum = 0.0_f64;
            let mut wv_sum = 0.0_f64;
            let mut count = 0usize;

            for i in 0..n {
                let dx = gx - obs_x[i];
                let dy = gy - obs_y[i];
                let d2 = dx * dx + dy * dy;

                if d2 > r2 {
                    continue;
                }

                // Coincident point -- return exact value.
                if d2 < 1e-30 {
                    return obs_values[i];
                }

                let w = match kind {
                    1 => {
                        // Barnes: w = exp(-d^2 / (kappa * gamma))
                        (-d2 / (kappa * gamma)).exp()
                    }
                    2 => {
                        // Cressman: w = (R^2 - d^2) / (R^2 + d^2)
                        (r2 - d2) / (r2 + d2)
                    }
                    _ => {
                        // Standard IDW: w = 1 / d^2
                        1.0 / d2
                    }
                };

                w_sum += w;
                wv_sum += w * obs_values[i];
                count += 1;
            }

            if count < min_neighbors {
                f64::NAN
            } else {
                wv_sum / w_sum
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idw_basic() {
        // Two equidistant points, target at midpoint => average.
        let obs_x = vec![0.0, 2.0];
        let obs_y = vec![0.0, 0.0];
        let obs_v = vec![10.0, 20.0];
        let gx = vec![1.0];
        let gy = vec![0.0];
        let out = inverse_distance_to_points(
            &obs_x, &obs_y, &obs_v, &gx, &gy,
            10.0, 1, 0, 100000.0, 0.2,
        );
        assert!((out[0] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_idw_coincident() {
        let obs_x = vec![5.0];
        let obs_y = vec![5.0];
        let obs_v = vec![42.0];
        let gx = vec![5.0];
        let gy = vec![5.0];
        let out = inverse_distance_to_points(
            &obs_x, &obs_y, &obs_v, &gx, &gy,
            10.0, 1, 0, 100000.0, 0.2,
        );
        assert!((out[0] - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_idw_too_few_neighbors() {
        let obs_x = vec![0.0];
        let obs_y = vec![0.0];
        let obs_v = vec![10.0];
        let gx = vec![100.0]; // outside radius
        let gy = vec![0.0];
        let out = inverse_distance_to_points(
            &obs_x, &obs_y, &obs_v, &gx, &gy,
            5.0, 1, 0, 100000.0, 0.2,
        );
        assert!(out[0].is_nan());
    }

    #[test]
    fn test_barnes_returns_value() {
        let obs_x = vec![0.0, 2.0];
        let obs_y = vec![0.0, 0.0];
        let obs_v = vec![10.0, 20.0];
        let gx = vec![1.0];
        let gy = vec![0.0];
        let out = inverse_distance_to_points(
            &obs_x, &obs_y, &obs_v, &gx, &gy,
            10.0, 1, 1, 100000.0, 0.2,
        );
        // With equal distances, Barnes should also give the average.
        assert!((out[0] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_cressman_returns_value() {
        let obs_x = vec![0.0, 2.0];
        let obs_y = vec![0.0, 0.0];
        let obs_v = vec![10.0, 20.0];
        let gx = vec![1.0];
        let gy = vec![0.0];
        let out = inverse_distance_to_points(
            &obs_x, &obs_y, &obs_v, &gx, &gy,
            10.0, 1, 2, 100000.0, 0.2,
        );
        // With equal distances, Cressman should also give the average.
        assert!((out[0] - 15.0).abs() < 1e-10);
    }
}
