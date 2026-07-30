use crate::CalcError;
use rustwx_core::GridShape;

pub fn sum_window_fields(grid: GridShape, fields: &[&[f64]]) -> Result<Vec<f64>, CalcError> {
    if fields.is_empty() {
        return Err(CalcError::EmptyWindowInputs { operation: "sum" });
    }

    let expected = grid.len();
    let mut out = vec![0.0; expected];
    for values in fields {
        if values.len() != expected {
            return Err(CalcError::LengthMismatch {
                field: "window_field",
                expected,
                actual: values.len(),
            });
        }
        for (target, value) in out.iter_mut().zip(values.iter()) {
            *target += *value;
        }
    }
    Ok(out)
}

pub fn max_window_fields(grid: GridShape, fields: &[&[f64]]) -> Result<Vec<f64>, CalcError> {
    if fields.is_empty() {
        return Err(CalcError::EmptyWindowInputs { operation: "max" });
    }

    let expected = grid.len();
    let mut out = vec![f64::NEG_INFINITY; expected];
    for values in fields {
        if values.len() != expected {
            return Err(CalcError::LengthMismatch {
                field: "window_field",
                expected,
                actual: values.len(),
            });
        }
        for (target, value) in out.iter_mut().zip(values.iter()) {
            *target = target.max(*value);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_window_fields_adds_all_inputs() {
        let grid = GridShape::new(2, 1).unwrap();
        let a = [1.0, 2.0];
        let b = [0.5, 0.25];
        let out = sum_window_fields(grid, &[&a, &b]).unwrap();
        assert_eq!(out, vec![1.5, 2.25]);
    }

    #[test]
    fn max_window_fields_uses_pointwise_maximum() {
        let grid = GridShape::new(2, 1).unwrap();
        let a = [1.0, 4.0];
        let b = [2.0, 3.0];
        let out = max_window_fields(grid, &[&a, &b]).unwrap();
        assert_eq!(out, vec![2.0, 4.0]);
    }

    #[test]
    fn window_reducers_reject_empty_input() {
        let grid = GridShape::new(1, 1).unwrap();
        let err = sum_window_fields(grid, &[]).unwrap_err();
        assert!(matches!(
            err,
            CalcError::EmptyWindowInputs { operation: "sum" }
        ));
    }
}
