use rustwx_render::theme::{air_quality_scale, aerosol_sequential_scale};
use rustwx_render::{ColormapBuildOptions, ColorScale, LegendMode, build_colormap};

#[test]
fn chem_palette_edges_equal_epa_concentration_breakpoints() {
    let pm = air_quality_scale(false);
    assert_eq!(pm.levels, vec![0.0, 9.0, 35.4, 55.4, 125.4, 225.4, 325.4]);
    let ozone = air_quality_scale(true);
    assert_eq!(ozone.levels, vec![0.0, 54.0, 70.0, 85.0, 105.0, 200.0]);
    assert_eq!(pm.colors.len(), pm.levels.len() - 1);
    assert_eq!(ozone.colors.len(), ozone.levels.len() - 1);
    assert_eq!(aerosol_sequential_scale(true).levels, vec![0.0, 0.1, 0.2, 0.5, 1.0, 2.0]);
    assert_eq!(aerosol_sequential_scale(false).levels,
               vec![0.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]);
}

#[test]
fn chem_threshold_colours_are_not_interpolated_across_unequal_bins() {
    let scale = air_quality_scale(false);
    let palette: Vec<_> = scale.colors.iter().copied().map(Into::into).collect();
    let mut options = ColormapBuildOptions::default();
    options.legend.mode = LegendMode::Thresholds;
    let map = build_colormap(&ColorScale::Discrete(scale.clone()), options);
    assert_eq!(map.levels, scale.levels);
    assert_eq!(map.colors, palette);
    assert!(!map.categories);
}
