use rustwx_render::{DiscreteColorScale, ExtendMode, WeatherPalette, palette_scale};

pub(crate) fn qpf_inches_scale() -> DiscreteColorScale {
    palette_scale(
        WeatherPalette::Precip,
        qpf_inches_levels(),
        ExtendMode::Max,
        Some(0.01),
    )
}

pub(crate) fn qpf_inches_levels() -> Vec<f64> {
    vec![
        0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60,
        1.80, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00, 6.00, 8.00, 10.00, 15.00,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qpf_inches_scale_keeps_standard_precip_family_breakpoints() {
        let scale = qpf_inches_scale();
        assert!(scale.levels.contains(&0.1));
        assert!(scale.levels.contains(&0.5));
        assert!(scale.levels.contains(&1.0));
        assert_eq!(scale.levels.last(), Some(&15.0));
        assert_eq!(scale.extend, ExtendMode::Max);
        assert_eq!(scale.mask_below, Some(0.01));
    }
}
