use crate::{finite, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EkiRequest {
    /// Member-major continuous parameter matrix. Categorical settings stay fixed.
    pub parameters: Vec<Vec<f64>>,
    /// Member-major training-only forecast features.
    pub predictions: Vec<Vec<f64>>,
    pub observations: Vec<f64>,
    pub variances: Vec<f64>,
    pub bounds: Vec<[f64; 2]>,
    pub damping: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct EkiResult {
    pub parameters: Vec<Vec<f64>>,
    pub clipped_values: usize,
    pub observation_features: usize,
    pub ensemble_members: usize,
    pub solve_dimension: usize,
    pub preupdate_whitened_residual_rms: Vec<f64>,
    pub method: &'static str,
    pub requires_new_forecasts: bool,
}

fn matrix_shape(matrix: &[Vec<f64>], rows: usize, columns: usize, name: &str) -> Result<()> {
    if matrix.len() != rows || matrix.iter().any(|row| row.len() != columns) {
        return Err(format!("{name} must have shape {rows} by {columns}"));
    }
    for row in matrix {
        for &value in row {
            finite(value, name)?;
        }
    }
    Ok(())
}

fn column_means(matrix: &[Vec<f64>], columns: usize) -> Result<Vec<f64>> {
    let mut means = vec![0.0; columns];
    for row in matrix {
        for (mean, value) in means.iter_mut().zip(row) {
            *mean = finite(*mean + *value / matrix.len() as f64, "ensemble mean")?;
        }
    }
    Ok(means)
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut factor = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i][j];
            for k in 0..j {
                value = finite(value - factor[i][k] * factor[j][k], "Cholesky residual")?;
            }
            if i == j {
                if value <= 0.0 {
                    return Err(
                        "EKI covariance solve is not positive definite at working precision".into(),
                    );
                }
                factor[i][j] = value.sqrt();
            } else {
                factor[i][j] = finite(value / factor[j][j], "Cholesky factor")?;
            }
        }
    }
    Ok(factor)
}

fn solve(factor: &[Vec<f64>], right: &[f64]) -> Result<Vec<f64>> {
    let n = factor.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut value = right[i];
        for j in 0..i {
            value = finite(value - factor[i][j] * result[j], "forward solve")?;
        }
        result[i] = finite(value / factor[i][i], "forward solve")?;
    }
    for i in (0..n).rev() {
        let mut value = result[i];
        for j in i + 1..n {
            value = finite(value - factor[j][i] * result[j], "back solve")?;
        }
        result[i] = finite(value / factor[i][i], "back solve")?;
    }
    Ok(result)
}

pub fn update(request: EkiRequest) -> Result<EkiResult> {
    let n = request.parameters.len();
    let p = request.bounds.len();
    let m = request.observations.len();
    if n < 2 || p == 0 || m == 0 {
        return Err(
            "EKI needs at least two members, one continuous parameter and one feature".into(),
        );
    }
    matrix_shape(&request.parameters, n, p, "parameters")?;
    matrix_shape(&request.predictions, n, m, "predictions")?;
    if request.variances.len() != m {
        return Err("one positive observation variance is required per feature".into());
    }
    finite(request.damping, "damping")?;
    if !(0.0..=1.0).contains(&request.damping) {
        return Err("damping must be between zero and one".into());
    }
    for j in 0..m {
        finite(request.observations[j], "observations")?;
        if finite(request.variances[j], "variances")? <= 0.0 {
            return Err("observation variances must be strictly positive".into());
        }
    }
    for (j, &[lo, hi]) in request.bounds.iter().enumerate() {
        finite(lo, "lower bound")?;
        finite(hi, "upper bound")?;
        if lo > hi {
            return Err("parameter lower bounds cannot exceed upper bounds".into());
        }
        if request
            .parameters
            .iter()
            .any(|row| row[j] < lo || row[j] > hi)
        {
            return Err(format!(
                "initial parameter column {j} is outside its bounds"
            ));
        }
    }
    let parameter_means = column_means(&request.parameters, p)?;
    let forecast_means = column_means(&request.predictions, m)?;
    let scale = 1.0 / ((n - 1) as f64).sqrt();
    let mut u = vec![vec![0.0; p]; n];
    let mut y = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..p {
            u[i][j] = finite(
                (request.parameters[i][j] - parameter_means[j]) * scale,
                "parameter anomaly",
            )?;
        }
        for j in 0..m {
            y[i][j] = finite(
                (request.predictions[i][j] - forecast_means[j]) * scale,
                "forecast anomaly",
            )?;
        }
    }
    // U and Y are sample-covariance anomalies. The Woodbury identity gives
    // U (I + Y' R^-1 Y)^-1 Y' R^-1 residual. Only nmember squared storage
    // is needed for the solve, even for many station-time features.
    let mut system = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in 0..=i {
            let mut value = if i == k { 1.0 } else { 0.0 };
            for j in 0..m {
                value = finite(
                    value + y[i][j] * (y[k][j] / request.variances[j]),
                    "ensemble-space covariance",
                )?;
            }
            system[i][k] = value;
            system[k][i] = value;
        }
    }
    let factor = cholesky(&system)?;
    let mut parameters = request.parameters.clone();
    let mut clipped_values = 0;
    let mut residual_rms = Vec::with_capacity(n);
    for member in 0..n {
        let mut right = vec![0.0; n];
        let mut residual_norm = 0.0;
        for j in 0..m {
            let residual = finite(
                request.observations[j] - request.predictions[member][j],
                "innovation",
            )?;
            let weighted = finite(residual / request.variances[j], "weighted innovation")?;
            residual_norm = finite(
                residual_norm + (residual * weighted) / m as f64,
                "whitened residual norm",
            )?;
            for i in 0..n {
                right[i] = finite(right[i] + y[i][j] * weighted, "ensemble-space innovation")?;
            }
        }
        residual_rms.push(residual_norm.max(0.0).sqrt());
        let weights = solve(&factor, &right)?;
        for j in 0..p {
            let mut delta = 0.0;
            for i in 0..n {
                delta = finite(delta + u[i][j] * weights[i], "parameter update")?;
            }
            let proposal = finite(
                parameters[member][j] + request.damping * delta,
                "updated parameter",
            )?;
            let clipped = proposal.clamp(request.bounds[j][0], request.bounds[j][1]);
            if clipped != proposal {
                clipped_values += 1;
            }
            parameters[member][j] = clipped;
        }
    }
    Ok(EkiResult {
        parameters,
        clipped_values,
        observation_features: m,
        ensemble_members: n,
        solve_dimension: n,
        preupdate_whitened_residual_rms: residual_rms,
        method: "deterministic_eki_ensemble_space_cholesky",
        requires_new_forecasts: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analytic() -> EkiRequest {
        EkiRequest {
            parameters: vec![vec![-1.0], vec![1.0]],
            predictions: vec![vec![-1.0], vec![1.0]],
            observations: vec![1.0],
            variances: vec![2.0],
            bounds: vec![[-10.0, 10.0]],
            damping: 1.0,
        }
    }

    #[test]
    fn linear_scalar_solution_matches_covariance_formula() {
        // Cuy = Cyy = 2; gain = 2 / (2 + 2) = 0.5.
        let result = update(analytic()).unwrap();
        assert!((result.parameters[0][0] - 0.0).abs() < 1e-12);
        assert!((result.parameters[1][0] - 1.0).abs() < 1e-12);
        assert_eq!(result.solve_dimension, 2);
        assert!(result.requires_new_forecasts);
    }

    #[test]
    fn multiple_features_match_analytic_gain() {
        let mut input = analytic();
        input.parameters = vec![vec![-1.0, -2.0], vec![1.0, 2.0]];
        input.predictions = vec![vec![-1.0, -2.0], vec![1.0, 2.0]];
        input.observations = vec![1.0, 2.0];
        input.variances = vec![2.0, 8.0];
        input.bounds = vec![[-10.0, 10.0], [-20.0, 20.0]];
        let result = update(input).unwrap();
        assert!((result.parameters[0][0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((result.parameters[0][1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((result.parameters[1][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bounds_damping_and_zero_covariance_are_respected() {
        let mut input = analytic();
        input.damping = 0.0;
        assert_eq!(update(input.clone()).unwrap().parameters, input.parameters);
        input.damping = 0.5;
        assert!((update(input).unwrap().parameters[0][0] + 0.5).abs() < 1e-12);
        let mut input = analytic();
        input.observations = vec![5.0];
        input.bounds = vec![[-1.0, 1.0]];
        let result = update(input).unwrap();
        assert_eq!(result.parameters, vec![vec![1.0], vec![1.0]]);
        assert_eq!(result.clipped_values, 2);
        let mut input = analytic();
        input.predictions = vec![vec![0.0], vec![0.0]];
        assert_eq!(update(input.clone()).unwrap().parameters, input.parameters);
    }

    #[test]
    fn invalid_shapes_variances_and_bounds_are_rejected() {
        let mut input = analytic();
        input.variances[0] = 0.0;
        assert!(update(input).unwrap_err().contains("positive"));
        let mut input = analytic();
        input.predictions[1].push(2.0);
        assert!(update(input).unwrap_err().contains("shape"));
        let mut input = analytic();
        input.parameters[0][0] = f64::INFINITY;
        assert!(update(input).unwrap_err().contains("finite"));
        let mut input = analytic();
        input.bounds[0] = [0.0, 1.0];
        assert!(update(input).unwrap_err().contains("outside"));
    }

    #[test]
    fn many_features_keep_member_sized_solve() {
        let mut input = analytic();
        input.predictions = vec![vec![-1.0; 2048], vec![1.0; 2048]];
        input.observations = vec![1.0; 2048];
        input.variances = vec![2.0; 2048];
        let result = update(input).unwrap();
        assert_eq!(result.solve_dimension, 2);
        let gain = 2048.0 / 2049.0;
        assert!((result.parameters[0][0] - (-1.0 + 2.0 * gain)).abs() < 1e-10);
    }
}
