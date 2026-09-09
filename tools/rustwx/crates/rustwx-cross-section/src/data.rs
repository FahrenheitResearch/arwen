use std::collections::BTreeMap;

use crate::error::CrossSectionError;
use crate::vertical::{VerticalAxis, VerticalKind, standard_atmosphere_height_m};

/// Lightweight metadata container for a cross-section field.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SectionMetadata {
    pub title: Option<String>,
    pub field_name: Option<String>,
    pub field_units: Option<String>,
    pub source: Option<String>,
    pub valid_label: Option<String>,
    pub attributes: BTreeMap<String, String>,
}

impl SectionMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn titled(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn field(mut self, field_name: impl Into<String>, field_units: impl Into<String>) -> Self {
        self.field_name = Some(field_name.into());
        self.field_units = Some(field_units.into());
        self
    }

    pub fn sourced_from(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn valid_at(mut self, valid_label: impl Into<String>) -> Self {
        self.valid_label = Some(valid_label.into());
        self
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    pub fn attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(String::as_str)
    }
}

/// Terrain or surface profile aligned to distance along the section.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainProfile {
    distances_km: Vec<f64>,
    surface_pressure_hpa: Option<Vec<f64>>,
    surface_height_m: Option<Vec<f64>>,
}

impl TerrainProfile {
    pub fn new(distances_km: Vec<f64>) -> Result<Self, CrossSectionError> {
        validate_distances(&distances_km)?;
        Ok(Self {
            distances_km,
            surface_pressure_hpa: None,
            surface_height_m: None,
        })
    }

    pub fn from_surface_pressure_hpa(
        distances_km: Vec<f64>,
        surface_pressure_hpa: Vec<f64>,
    ) -> Result<Self, CrossSectionError> {
        Self::new(distances_km)?.with_surface_pressure_hpa(surface_pressure_hpa)
    }

    pub fn from_surface_height_m(
        distances_km: Vec<f64>,
        surface_height_m: Vec<f64>,
    ) -> Result<Self, CrossSectionError> {
        Self::new(distances_km)?.with_surface_height_m(surface_height_m)
    }

    pub fn with_surface_pressure_hpa(
        mut self,
        surface_pressure_hpa: Vec<f64>,
    ) -> Result<Self, CrossSectionError> {
        validate_profile_values(&self.distances_km, &surface_pressure_hpa, true)?;
        self.surface_pressure_hpa = Some(surface_pressure_hpa);
        Ok(self)
    }

    pub fn with_surface_height_m(
        mut self,
        surface_height_m: Vec<f64>,
    ) -> Result<Self, CrossSectionError> {
        validate_profile_values(&self.distances_km, &surface_height_m, false)?;
        self.surface_height_m = Some(surface_height_m);
        Ok(self)
    }

    pub fn distances_km(&self) -> &[f64] {
        &self.distances_km
    }

    pub fn surface_pressure_hpa(&self) -> Option<&[f64]> {
        self.surface_pressure_hpa.as_deref()
    }

    pub fn surface_height_m(&self) -> Option<&[f64]> {
        self.surface_height_m.as_deref()
    }

    pub fn surface_pressure_at(&self, distance_km: f64) -> Option<f64> {
        interpolate_profile(
            &self.distances_km,
            self.surface_pressure_hpa.as_deref()?,
            distance_km,
        )
    }

    pub fn surface_height_m_at(&self, distance_km: f64) -> Option<f64> {
        interpolate_profile(
            &self.distances_km,
            self.surface_height_m.as_deref()?,
            distance_km,
        )
    }

    pub fn surface_value_on_axis(&self, axis: &VerticalAxis, distance_km: f64) -> Option<f64> {
        match axis.kind() {
            VerticalKind::Pressure => self.surface_pressure_at(distance_km),
            VerticalKind::Height => {
                if let Some(height_m) = self.surface_height_m_at(distance_km) {
                    axis.convert_height_m_to_axis_units(height_m)
                } else {
                    let pressure_hpa = self.surface_pressure_at(distance_km)?;
                    axis.convert_height_m_to_axis_units(standard_atmosphere_height_m(pressure_hpa))
                }
            }
        }
    }

    pub fn below_surface(
        &self,
        axis: &VerticalAxis,
        distance_km: f64,
        axis_value: f64,
    ) -> Option<bool> {
        let surface_value = self.surface_value_on_axis(axis, distance_km)?;
        Some(match axis.kind() {
            VerticalKind::Pressure => axis_value > surface_value,
            VerticalKind::Height => axis_value < surface_value,
        })
    }
}

/// A scalar cross-section field stored as `[level][point]` row-major data.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarSection {
    distances_km: Vec<f64>,
    vertical_axis: VerticalAxis,
    values: Vec<f32>,
    metadata: SectionMetadata,
    terrain: Option<TerrainProfile>,
}

impl ScalarSection {
    pub fn new(
        distances_km: Vec<f64>,
        vertical_axis: VerticalAxis,
        values: Vec<f32>,
    ) -> Result<Self, CrossSectionError> {
        validate_distances(&distances_km)?;
        let expected = distances_km.len() * vertical_axis.len();
        if expected != values.len() {
            return Err(CrossSectionError::ShapeMismatch {
                context: "scalar section values",
                expected,
                actual: values.len(),
            });
        }

        Ok(Self {
            distances_km,
            vertical_axis,
            values,
            metadata: SectionMetadata::default(),
            terrain: None,
        })
    }

    pub fn with_metadata(mut self, metadata: SectionMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_terrain(mut self, terrain: TerrainProfile) -> Result<Self, CrossSectionError> {
        if terrain.distances_km().len() < 2 {
            return Err(CrossSectionError::InvalidTerrainProfile);
        }
        self.terrain = Some(terrain);
        Ok(self)
    }

    pub fn distances_km(&self) -> &[f64] {
        &self.distances_km
    }

    pub fn vertical_axis(&self) -> &VerticalAxis {
        &self.vertical_axis
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn metadata(&self) -> &SectionMetadata {
        &self.metadata
    }

    pub fn terrain(&self) -> Option<&TerrainProfile> {
        self.terrain.as_ref()
    }

    pub fn n_levels(&self) -> usize {
        self.vertical_axis.len()
    }

    pub fn n_points(&self) -> usize {
        self.distances_km.len()
    }

    pub fn value(&self, level_index: usize, point_index: usize) -> Option<f32> {
        let n_points = self.n_points();
        if level_index >= self.n_levels() || point_index >= n_points {
            return None;
        }
        Some(self.values[level_index * n_points + point_index])
    }

    pub fn finite_range(&self) -> Option<(f32, f32)> {
        let mut min_value = f32::INFINITY;
        let mut max_value = f32::NEG_INFINITY;
        let mut any = false;

        for value in &self.values {
            if value.is_finite() {
                min_value = min_value.min(*value);
                max_value = max_value.max(*value);
                any = true;
            }
        }

        any.then_some((min_value, max_value))
    }

    pub fn bilinear_sample(&self, distance_km: f64, axis_value: f64) -> Option<f32> {
        let x_index = fractional_index_increasing(&self.distances_km, distance_km)?;
        let y_index = self.vertical_axis.fractional_index(axis_value)?;
        self.bilinear_sample_at_indices(x_index, y_index)
    }

    pub fn masked_with_terrain(&self) -> Self {
        let Some(terrain) = self.terrain.as_ref() else {
            return self.clone();
        };

        let mut masked_values = self.values.clone();
        let n_points = self.n_points();
        for (point_index, &distance_km) in self.distances_km.iter().enumerate() {
            for (level_index, &axis_value) in self.vertical_axis.levels().iter().enumerate() {
                let below_ground = terrain
                    .below_surface(&self.vertical_axis, distance_km, axis_value)
                    .unwrap_or(false);
                if below_ground {
                    masked_values[level_index * n_points + point_index] = f32::NAN;
                }
            }
        }

        Self {
            distances_km: self.distances_km.clone(),
            vertical_axis: self.vertical_axis.clone(),
            values: masked_values,
            metadata: self.metadata.clone(),
            terrain: self.terrain.clone(),
        }
    }

    /// Bilinear sample that survives a partly-missing quad: the finite
    /// corners keep their bilinear weights, the missing ones give theirs
    /// up, and the second value reports how much of the quad was finite.
    ///
    /// WHAT BREAKAGE THIS PREVENTS (gate law): the strict
    /// [`Self::bilinear_sample`] drops the WHOLE quad the moment one corner
    /// is NaN.  On a `~log` fill, whose floor IS a NaN, and on any field
    /// that ends inside the domain, that turned a smooth plume edge into a
    /// staircase of whole sample cells -- the delivered sections read as
    /// per-column blocks even though the data under them is 1 km.  Fading
    /// the edge by its coverage draws the same field with no staircase
    /// (gpuwm addition, VENDOR.md).
    pub fn bilinear_sample_coverage(
        &self,
        distance_km: f64,
        axis_value: f64,
    ) -> Option<(f32, f32)> {
        let x_index = fractional_index_increasing(&self.distances_km, distance_km)?;
        let y_index = self.vertical_axis.fractional_index(axis_value)?;
        let n_points = self.n_points();
        let n_levels = self.n_levels();
        if n_points == 0 || n_levels == 0 {
            return None;
        }

        let x0 = x_index.floor().clamp(0.0, (n_points - 1) as f64) as usize;
        let x1 = x_index.ceil().clamp(0.0, (n_points - 1) as f64) as usize;
        let y0 = y_index.floor().clamp(0.0, (n_levels - 1) as f64) as usize;
        let y1 = y_index.ceil().clamp(0.0, (n_levels - 1) as f64) as usize;

        let fx = (x_index - x0 as f64) as f32;
        let fy = (y_index - y0 as f64) as f32;

        let corners = [
            (self.value(y0, x0), (1.0 - fx) * (1.0 - fy)),
            (self.value(y0, x1), fx * (1.0 - fy)),
            (self.value(y1, x0), (1.0 - fx) * fy),
            (self.value(y1, x1), fx * fy),
        ];

        let mut sum = 0.0f32;
        let mut weight = 0.0f32;
        for (value, w) in corners {
            let Some(value) = value else { continue };
            if !value.is_finite() || w <= 0.0 {
                continue;
            }
            sum += value * w;
            weight += w;
        }
        if weight <= 0.0 {
            return None;
        }
        Some((sum / weight, weight.clamp(0.0, 1.0)))
    }

    fn bilinear_sample_at_indices(&self, x_index: f64, y_index: f64) -> Option<f32> {
        let n_points = self.n_points();
        let n_levels = self.n_levels();

        if n_points == 0 || n_levels == 0 {
            return None;
        }

        let x0 = x_index.floor().clamp(0.0, (n_points - 1) as f64) as usize;
        let x1 = x_index.ceil().clamp(0.0, (n_points - 1) as f64) as usize;
        let y0 = y_index.floor().clamp(0.0, (n_levels - 1) as f64) as usize;
        let y1 = y_index.ceil().clamp(0.0, (n_levels - 1) as f64) as usize;

        let fx = (x_index - x0 as f64) as f32;
        let fy = (y_index - y0 as f64) as f32;

        let v00 = self.value(y0, x0)?;
        let v10 = self.value(y0, x1)?;
        let v01 = self.value(y1, x0)?;
        let v11 = self.value(y1, x1)?;
        if !(v00.is_finite() && v10.is_finite() && v01.is_finite() && v11.is_finite()) {
            return None;
        }

        let top = v00 * (1.0 - fx) + v10 * fx;
        let bottom = v01 * (1.0 - fx) + v11 * fx;
        Some(top * (1.0 - fy) + bottom * fy)
    }
}

pub(crate) fn fractional_index_increasing(values: &[f64], target: f64) -> Option<f64> {
    if values.len() < 2 || !target.is_finite() {
        return None;
    }
    if target < values[0] || target > values[values.len() - 1] {
        return None;
    }

    if (target - values[0]).abs() <= f64::EPSILON {
        return Some(0.0);
    }
    if (target - values[values.len() - 1]).abs() <= f64::EPSILON {
        return Some((values.len() - 1) as f64);
    }

    for (index, pair) in values.windows(2).enumerate() {
        if target >= pair[0] && target <= pair[1] {
            let fraction = (target - pair[0]) / (pair[1] - pair[0]);
            return Some(index as f64 + fraction);
        }
    }

    None
}

fn interpolate_profile(xs: &[f64], ys: &[f64], x: f64) -> Option<f64> {
    let x_index = fractional_index_increasing(xs, x.clamp(xs[0], xs[xs.len() - 1]))?;
    let i0 = x_index.floor() as usize;
    let i1 = x_index.ceil() as usize;
    if i0 == i1 {
        return ys.get(i0).copied();
    }
    let fraction = x_index - i0 as f64;
    let y0 = *ys.get(i0)?;
    let y1 = *ys.get(i1)?;
    Some(y0 + fraction * (y1 - y0))
}

fn validate_distances(distances_km: &[f64]) -> Result<(), CrossSectionError> {
    if distances_km.len() < 2 {
        return Err(CrossSectionError::InvalidSampleCount);
    }
    if distances_km.iter().any(|distance| !distance.is_finite()) {
        return Err(CrossSectionError::NonMonotonicDistances);
    }
    if !distances_km
        .windows(2)
        .all(|pair| pair[1] > pair[0] && pair[0].is_finite() && pair[1].is_finite())
    {
        return Err(CrossSectionError::NonMonotonicDistances);
    }
    Ok(())
}

fn validate_profile_values(
    distances_km: &[f64],
    values: &[f64],
    positive_only: bool,
) -> Result<(), CrossSectionError> {
    if distances_km.len() != values.len() || values.len() < 2 {
        return Err(CrossSectionError::InvalidTerrainProfile);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CrossSectionError::InvalidTerrainProfile);
    }
    if positive_only && values.iter().any(|value| *value <= 0.0) {
        return Err(CrossSectionError::InvalidTerrainProfile);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertical::VerticalAxis;

    #[test]
    fn terrain_mask_replaces_subsurface_values_with_nan() {
        let axis = VerticalAxis::pressure_hpa(vec![1000.0, 900.0, 800.0]).unwrap();
        let section = ScalarSection::new(
            vec![0.0, 10.0],
            axis,
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0,
            ],
        )
        .unwrap()
        .with_terrain(
            TerrainProfile::from_surface_pressure_hpa(vec![0.0, 10.0], vec![950.0, 850.0]).unwrap(),
        )
        .unwrap();

        let masked = section.masked_with_terrain();
        assert!(masked.value(0, 0).unwrap().is_nan());
        assert!(masked.value(0, 1).unwrap().is_nan());
        assert!(masked.value(1, 1).unwrap().is_nan());
        assert_eq!(masked.value(2, 0).unwrap(), 5.0);
    }

    #[test]
    fn terrain_profile_can_project_pressure_to_height_axis() {
        let terrain =
            TerrainProfile::from_surface_pressure_hpa(vec![0.0, 100.0], vec![1000.0, 800.0])
                .unwrap();
        let axis = VerticalAxis::height_km(vec![0.0, 1.0, 2.0, 4.0]).unwrap();

        let surface0 = terrain.surface_value_on_axis(&axis, 0.0).unwrap();
        let surface1 = terrain.surface_value_on_axis(&axis, 100.0).unwrap();
        assert!(surface1 > surface0);
    }
}
