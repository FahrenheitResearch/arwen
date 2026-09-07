use crate::error::CrossSectionError;

/// Logical vertical coordinate type used by a cross-section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalKind {
    Pressure,
    Height,
}

/// Units associated with a [`VerticalAxis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalUnits {
    Hectopascals,
    Meters,
    Kilometers,
}

/// Plot transform for the vertical axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalScale {
    Linear,
    Logarithmic,
}

/// A monotonic vertical axis with enough information to map between data values and plot space.
#[derive(Debug, Clone, PartialEq)]
pub struct VerticalAxis {
    levels: Vec<f64>,
    kind: VerticalKind,
    units: VerticalUnits,
    scale: VerticalScale,
}

impl VerticalAxis {
    pub fn pressure_hpa(levels: Vec<f64>) -> Result<Self, CrossSectionError> {
        Self::from_levels(
            levels,
            VerticalKind::Pressure,
            VerticalUnits::Hectopascals,
            VerticalScale::Logarithmic,
        )
    }

    pub fn height_meters(levels: Vec<f64>) -> Result<Self, CrossSectionError> {
        Self::from_levels(
            levels,
            VerticalKind::Height,
            VerticalUnits::Meters,
            VerticalScale::Linear,
        )
    }

    pub fn height_km(levels: Vec<f64>) -> Result<Self, CrossSectionError> {
        Self::from_levels(
            levels,
            VerticalKind::Height,
            VerticalUnits::Kilometers,
            VerticalScale::Linear,
        )
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    pub fn kind(&self) -> VerticalKind {
        self.kind
    }

    pub fn units(&self) -> VerticalUnits {
        self.units
    }

    pub fn scale(&self) -> VerticalScale {
        self.scale
    }

    pub fn plot_top(&self) -> f64 {
        match self.kind {
            VerticalKind::Pressure => self.min_level(),
            VerticalKind::Height => self.max_level(),
        }
    }

    pub fn plot_bottom(&self) -> f64 {
        match self.kind {
            VerticalKind::Pressure => self.max_level(),
            VerticalKind::Height => self.min_level(),
        }
    }

    pub fn fractional_index(&self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }

        let min_level = self.min_level();
        let max_level = self.max_level();
        if value < min_level || value > max_level {
            return None;
        }

        if (value - self.levels[0]).abs() <= f64::EPSILON {
            return Some(0.0);
        }
        if (value - self.levels[self.levels.len() - 1]).abs() <= f64::EPSILON {
            return Some((self.levels.len() - 1) as f64);
        }

        for (index, pair) in self.levels.windows(2).enumerate() {
            let start = pair[0];
            let end = pair[1];
            let lo = start.min(end);
            let hi = start.max(end);
            if value >= lo && value <= hi {
                let fraction = if (end - start).abs() <= f64::EPSILON {
                    0.0
                } else {
                    (value - start) / (end - start)
                };
                return Some(index as f64 + fraction);
            }
        }

        None
    }

    pub fn plot_fraction_of_value(&self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }

        let min_level = self.min_level();
        let max_level = self.max_level();
        if value < min_level || value > max_level {
            return None;
        }

        let top = self.plot_top();
        let bottom = self.plot_bottom();
        let fraction = match self.scale {
            VerticalScale::Linear => (value - top) / (bottom - top),
            VerticalScale::Logarithmic => {
                if value <= 0.0 || top <= 0.0 || bottom <= 0.0 {
                    return None;
                }
                (value.ln() - top.ln()) / (bottom.ln() - top.ln())
            }
        };

        Some(fraction)
    }

    pub fn value_at_plot_fraction(&self, fraction: f64) -> f64 {
        let fraction = fraction.clamp(0.0, 1.0);
        let top = self.plot_top();
        let bottom = self.plot_bottom();
        match self.scale {
            VerticalScale::Linear => top + fraction * (bottom - top),
            VerticalScale::Logarithmic => (top.ln() + fraction * (bottom.ln() - top.ln())).exp(),
        }
    }

    pub fn pixel_y(&self, value: f64, plot_height: u32) -> Option<f64> {
        if plot_height < 2 {
            return None;
        }
        let fraction = self.plot_fraction_of_value(value)?;
        Some(fraction * (plot_height as f64 - 1.0))
    }

    pub(crate) fn convert_height_m_to_axis_units(&self, meters: f64) -> Option<f64> {
        if !meters.is_finite() {
            return None;
        }
        match self.units {
            VerticalUnits::Meters => Some(meters),
            VerticalUnits::Kilometers => Some(meters / 1_000.0),
            VerticalUnits::Hectopascals => None,
        }
    }

    fn from_levels(
        levels: Vec<f64>,
        kind: VerticalKind,
        units: VerticalUnits,
        scale: VerticalScale,
    ) -> Result<Self, CrossSectionError> {
        if levels.len() < 2 {
            return Err(CrossSectionError::EmptyLevels);
        }
        if levels.iter().any(|level| !level.is_finite()) {
            return Err(CrossSectionError::InvalidLevelValue);
        }
        if matches!(scale, VerticalScale::Logarithmic) && levels.iter().any(|level| *level <= 0.0) {
            return Err(CrossSectionError::InvalidLevelValue);
        }

        let mut ordering = 0i8;
        for pair in levels.windows(2) {
            let delta = pair[1] - pair[0];
            if delta.abs() <= f64::EPSILON {
                return Err(CrossSectionError::NonMonotonicLevels);
            }
            let sign = if delta > 0.0 { 1 } else { -1 };
            if ordering == 0 {
                ordering = sign;
            } else if ordering != sign {
                return Err(CrossSectionError::NonMonotonicLevels);
            }
        }

        Ok(Self {
            levels,
            kind,
            units,
            scale,
        })
    }

    fn min_level(&self) -> f64 {
        self.levels
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, level| acc.min(level))
    }

    fn max_level(&self) -> f64 {
        self.levels
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |acc, level| acc.max(level))
    }
}

/// Converts pressure in hPa to an approximate geometric height in meters using a
/// standard-atmosphere relationship.
pub fn standard_atmosphere_height_m(pressure_hpa: f64) -> f64 {
    44_330.0 * (1.0 - (pressure_hpa / 1_013.25).powf(0.1903))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_axis_maps_surface_below_upper_air() {
        let axis = VerticalAxis::pressure_hpa(vec![1000.0, 850.0, 700.0, 500.0]).unwrap();
        let surface = axis.pixel_y(1000.0, 200).unwrap();
        let mid = axis.pixel_y(700.0, 200).unwrap();
        let top = axis.pixel_y(500.0, 200).unwrap();

        assert!(surface > mid);
        assert!(mid > top);
        assert_eq!(axis.fractional_index(700.0).unwrap(), 2.0);
    }

    #[test]
    fn height_axis_uses_linear_scaling() {
        let axis = VerticalAxis::height_km(vec![0.0, 1.0, 2.0, 4.0, 8.0]).unwrap();
        let bottom = axis.pixel_y(0.0, 100).unwrap();
        let top = axis.pixel_y(8.0, 100).unwrap();
        let midpoint = axis.pixel_y(4.0, 100).unwrap();

        assert!(bottom > midpoint);
        assert!(midpoint > top);
        assert!((axis.value_at_plot_fraction(0.5) - 4.0).abs() < 1e-6);
    }
}
