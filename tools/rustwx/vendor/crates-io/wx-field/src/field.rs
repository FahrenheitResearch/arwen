/// 2D gridded weather field — the core data structure for model output.

use crate::error::{WxFieldError, Result};
use crate::meta::FieldMeta;
use crate::projection::Projection;

/// A 2D gridded field of floating-point values on a map projection.
///
/// This is the primary data container for model output (temperature, wind,
/// reflectivity, etc.). Data is stored in row-major order (west-to-east,
/// then south-to-north).
#[derive(Debug)]
pub struct Field2D {
    /// Flat data array in row-major order (length = nx * ny).
    pub data: Vec<f64>,
    /// Number of grid points in the x (west-east) direction.
    pub nx: usize,
    /// Number of grid points in the y (south-north) direction.
    pub ny: usize,
    /// Map projection for this field.
    pub projection: Box<dyn Projection>,
    /// Field metadata (variable name, units, level, etc.).
    pub meta: FieldMeta,
}

// Field2D contains Box<dyn Projection> which requires Send + Sync.
// The Projection trait already requires Send + Sync, so this is safe.
unsafe impl Send for Field2D {}
unsafe impl Sync for Field2D {}

impl Field2D {
    /// Create a new Field2D from data, dimensions, projection, and metadata.
    ///
    /// Returns an error if data.len() != nx * ny.
    pub fn new(
        data: Vec<f64>,
        nx: usize,
        ny: usize,
        projection: Box<dyn Projection>,
        meta: FieldMeta,
    ) -> Result<Self> {
        if data.len() != nx * ny {
            return Err(WxFieldError::InvalidDimensions(
                format!("data length {} != nx*ny ({}*{}={})", data.len(), nx, ny, nx * ny),
            ));
        }
        if nx == 0 || ny == 0 {
            return Err(WxFieldError::InvalidDimensions(
                format!("nx and ny must be > 0, got nx={} ny={}", nx, ny),
            ));
        }
        Ok(Self { data, nx, ny, projection, meta })
    }

    /// Get value at grid point (i, j). Returns None if out of bounds.
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        if i < self.nx && j < self.ny {
            Some(self.data[j * self.nx + i])
        } else {
            None
        }
    }

    /// Set value at grid point (i, j). Returns false if out of bounds.
    pub fn set(&mut self, i: usize, j: usize, value: f64) -> bool {
        if i < self.nx && j < self.ny {
            self.data[j * self.nx + i] = value;
            true
        } else {
            false
        }
    }

    /// Get the (lat, lon) for a grid point.
    pub fn latlon_at(&self, i: usize, j: usize) -> (f64, f64) {
        self.projection.grid_to_latlon(i as f64, j as f64)
    }

    /// Find the grid (i, j) for a given (lat, lon).
    pub fn grid_at(&self, lat: f64, lon: f64) -> (f64, f64) {
        self.projection.latlon_to_grid(lat, lon)
    }

    /// Minimum value in the field, ignoring NaN.
    pub fn min(&self) -> f64 {
        self.data.iter().copied().filter(|v| !v.is_nan()).fold(f64::INFINITY, f64::min)
    }

    /// Maximum value in the field, ignoring NaN.
    pub fn max(&self) -> f64 {
        self.data.iter().copied().filter(|v| !v.is_nan()).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Total number of grid points.
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns true if the field has zero grid points (should never happen after construction).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::Units;
    use crate::projection::LatLonProjection;

    fn test_field() -> Field2D {
        let proj = LatLonProjection::new(20.0, -130.0, 55.0, -60.0, 10, 5);
        let meta = FieldMeta::new("TMP", Units::Kelvin);
        let data: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();
        Field2D::new(data, 10, 5, Box::new(proj), meta).unwrap()
    }

    #[test]
    fn test_field_creation() {
        let field = test_field();
        assert_eq!(field.nx, 10);
        assert_eq!(field.ny, 5);
        assert_eq!(field.len(), 50);
    }

    #[test]
    fn test_field_get_set() {
        let mut field = test_field();
        assert_eq!(field.get(0, 0), Some(0.0));
        assert_eq!(field.get(9, 4), Some(24.5));
        assert_eq!(field.get(10, 0), None);

        assert!(field.set(0, 0, 999.0));
        assert_eq!(field.get(0, 0), Some(999.0));
        assert!(!field.set(10, 0, 1.0));
    }

    #[test]
    fn test_field_min_max() {
        let field = test_field();
        assert!((field.min() - 0.0).abs() < 1e-10);
        assert!((field.max() - 24.5).abs() < 1e-10);
    }

    #[test]
    fn test_field_dimension_mismatch() {
        let proj = LatLonProjection::new(20.0, -130.0, 55.0, -60.0, 10, 5);
        let meta = FieldMeta::new("TMP", Units::Kelvin);
        let data = vec![0.0; 49]; // wrong length
        let result = Field2D::new(data, 10, 5, Box::new(proj), meta);
        assert!(result.is_err());
    }

    #[test]
    fn test_field_zero_dimensions() {
        let proj = LatLonProjection::new(20.0, -130.0, 55.0, -60.0, 10, 5);
        let meta = FieldMeta::new("TMP", Units::Kelvin);
        let result = Field2D::new(vec![], 0, 5, Box::new(proj), meta);
        assert!(result.is_err());
    }
}
