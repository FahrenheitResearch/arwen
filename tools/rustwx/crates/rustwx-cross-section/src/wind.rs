use crate::error::CrossSectionError;

/// Wind resolved relative to the cross-section orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindDecomposition {
    pub along_section_ms: f64,
    pub left_of_section_ms: f64,
    pub speed_ms: f64,
}

/// Along/across-section wind fields stored as `[level][point]` row-major arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct DecomposedWindGrid {
    along_section_ms: Vec<f32>,
    left_of_section_ms: Vec<f32>,
    speed_ms: Vec<f32>,
    n_levels: usize,
    n_points: usize,
}

impl DecomposedWindGrid {
    pub fn along_section_ms(&self) -> &[f32] {
        &self.along_section_ms
    }

    pub fn left_of_section_ms(&self) -> &[f32] {
        &self.left_of_section_ms
    }

    pub fn speed_ms(&self) -> &[f32] {
        &self.speed_ms
    }

    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn along_section_value(&self, level_index: usize, point_index: usize) -> Option<f32> {
        self.flat_index(level_index, point_index)
            .map(|index| self.along_section_ms[index])
    }

    pub fn left_of_section_value(&self, level_index: usize, point_index: usize) -> Option<f32> {
        self.flat_index(level_index, point_index)
            .map(|index| self.left_of_section_ms[index])
    }

    pub fn speed_value(&self, level_index: usize, point_index: usize) -> Option<f32> {
        self.flat_index(level_index, point_index)
            .map(|index| self.speed_ms[index])
    }

    fn flat_index(&self, level_index: usize, point_index: usize) -> Option<usize> {
        if level_index >= self.n_levels || point_index >= self.n_points {
            return None;
        }
        Some(level_index * self.n_points + point_index)
    }
}

/// Decomposes a single eastward/northward wind vector relative to a section bearing in degrees.
///
/// Positive `along_section_ms` points toward the section end. Positive `left_of_section_ms`
/// points to the left when looking from the start point toward the end point.
pub fn decompose_wind(u_ms: f64, v_ms: f64, section_bearing_deg: f64) -> WindDecomposition {
    let theta = section_bearing_deg.to_radians();
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();

    let along = u_ms * sin_theta + v_ms * cos_theta;
    let left = -u_ms * cos_theta + v_ms * sin_theta;
    WindDecomposition {
        along_section_ms: along,
        left_of_section_ms: left,
        speed_ms: (u_ms * u_ms + v_ms * v_ms).sqrt(),
    }
}

/// Decomposes section-wide eastward and northward wind arrays into along/across components.
pub fn decompose_wind_grid(
    u_ms: &[f32],
    v_ms: &[f32],
    n_levels: usize,
    n_points: usize,
    section_bearings_deg: &[f64],
) -> Result<DecomposedWindGrid, CrossSectionError> {
    if n_levels < 1 {
        return Err(CrossSectionError::EmptyLevels);
    }
    if n_points < 2 {
        return Err(CrossSectionError::InvalidSampleCount);
    }
    let expected = n_levels * n_points;
    if u_ms.len() != expected {
        return Err(CrossSectionError::ShapeMismatch {
            context: "u wind grid",
            expected,
            actual: u_ms.len(),
        });
    }
    if v_ms.len() != expected {
        return Err(CrossSectionError::ShapeMismatch {
            context: "v wind grid",
            expected,
            actual: v_ms.len(),
        });
    }
    if section_bearings_deg.len() != n_points {
        return Err(CrossSectionError::ShapeMismatch {
            context: "section bearings",
            expected: n_points,
            actual: section_bearings_deg.len(),
        });
    }
    if section_bearings_deg
        .iter()
        .any(|bearing| !bearing.is_finite())
    {
        return Err(CrossSectionError::InvalidCoordinate);
    }

    let mut along_section_ms = vec![0.0f32; expected];
    let mut left_of_section_ms = vec![0.0f32; expected];
    let mut speed_ms = vec![0.0f32; expected];

    for level_index in 0..n_levels {
        for point_index in 0..n_points {
            let flat_index = level_index * n_points + point_index;
            let u = u_ms[flat_index];
            let v = v_ms[flat_index];
            if !(u.is_finite() && v.is_finite()) {
                along_section_ms[flat_index] = f32::NAN;
                left_of_section_ms[flat_index] = f32::NAN;
                speed_ms[flat_index] = f32::NAN;
                continue;
            }

            let resolved = decompose_wind(u as f64, v as f64, section_bearings_deg[point_index]);
            along_section_ms[flat_index] = resolved.along_section_ms as f32;
            left_of_section_ms[flat_index] = resolved.left_of_section_ms as f32;
            speed_ms[flat_index] = resolved.speed_ms as f32;
        }
    }

    Ok(DecomposedWindGrid {
        along_section_ms,
        left_of_section_ms,
        speed_ms,
        n_levels,
        n_points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn east_west_section_maps_eastward_wind_to_along_component() {
        let resolved = decompose_wind(10.0, 3.0, 90.0);
        assert!((resolved.along_section_ms - 10.0).abs() < 1e-6);
        assert!((resolved.left_of_section_ms - 3.0).abs() < 1e-6);
    }

    #[test]
    fn grid_decomposition_uses_column_bearings() {
        let grid = decompose_wind_grid(
            &[10.0, 0.0, 10.0, 0.0],
            &[0.0, 10.0, 0.0, 10.0],
            2,
            2,
            &[90.0, 0.0],
        )
        .unwrap();

        assert_eq!(grid.along_section_ms()[0], 10.0);
        assert_eq!(grid.along_section_ms()[1], 10.0);
        assert_eq!(grid.along_section_ms()[2], 10.0);
        assert_eq!(grid.along_section_ms()[3], 10.0);
    }
}
