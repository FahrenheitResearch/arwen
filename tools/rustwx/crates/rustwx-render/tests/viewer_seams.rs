//! Pins for the viewer-facing seams: `build_colormap`, `colorbar_ticks`,
//! `legend_color_at_rel`, `legend_tick_rel`, and `format_tick` are re-exports
//! of the exact code the PNG render path runs, so an external legend widget
//! (the egui data viewer) can reproduce production colors and tick values.

use rustwx_render::{
    Color, ColorScale, ColormapBuildOptions, DiscreteColorScale, ExtendMode, LegendControls,
    LegendMode, LevelDensity, RenderDensity, Rgba, build_colormap, colorbar_ticks, format_tick,
    legend_color_at_rel, legend_tick_rel,
};

fn two_bin_scale() -> ColorScale {
    ColorScale::Discrete(DiscreteColorScale {
        levels: vec![0.0, 10.0, 20.0],
        colors: vec![Color::rgba(10, 20, 30, 255), Color::rgba(200, 100, 50, 255)],
        extend: ExtendMode::Neither,
        mask_below: Some(0.0),
    })
}

/// No fill/palette densification — the reference-discrete options the heavy
/// lane renders with, where interval colors equal the listed scale colors.
fn reference_options() -> ColormapBuildOptions {
    ColormapBuildOptions {
        render_density: RenderDensity {
            fill: LevelDensity::default(),
            palette_multiplier: 1,
        },
        legend: LegendControls::default(),
    }
}

#[test]
fn build_colormap_maps_values_to_interval_colors_and_masks() {
    let cmap = build_colormap(&two_bin_scale(), reference_options());
    assert_eq!(cmap.map(5.0), Rgba::with_alpha(10, 20, 30, 255));
    assert_eq!(cmap.map(15.0), Rgba::with_alpha(200, 100, 50, 255));
    // NaN and below-mask values are transparent — the viewer must honor that.
    assert_eq!(cmap.map(f64::NAN), Rgba::TRANSPARENT);
    assert_eq!(cmap.map(-1.0), Rgba::TRANSPARENT);
    // Extend::Neither: below the first level (but >= mask) is transparent too.
    assert_eq!(cmap.legend_levels_for_display(), &[0.0, 10.0, 20.0]);
}

#[test]
fn colorbar_ticks_honors_explicit_step_and_auto_nice_steps() {
    let cmap = build_colormap(&two_bin_scale(), ColormapBuildOptions::default());
    // Densification never changes the legend levels ticks derive from.
    assert_eq!(colorbar_ticks(&cmap, Some(10.0)), vec![0.0, 10.0, 20.0]);
    // Auto step picks a "nice" subdivision covering the level range.
    let auto = colorbar_ticks(&cmap, None);
    assert_eq!(auto.first().copied(), Some(0.0));
    assert_eq!(auto.last().copied(), Some(20.0));
    assert!(auto.len() >= 3);
}

#[test]
fn legend_sampling_matches_interval_colors_and_value_linear_ticks() {
    let cmap = build_colormap(&two_bin_scale(), reference_options());
    // Stepped: first half of the bar is the first interval color.
    assert_eq!(
        legend_color_at_rel(&cmap, LegendMode::Stepped, 0.25),
        Rgba::with_alpha(10, 20, 30, 255)
    );
    assert_eq!(
        legend_color_at_rel(&cmap, LegendMode::Stepped, 0.75),
        Rgba::with_alpha(200, 100, 50, 255)
    );
    // Tick positions are linear by value across the legend range.
    assert_eq!(legend_tick_rel(&cmap, 0.0), Some(0.0));
    assert_eq!(legend_tick_rel(&cmap, 10.0), Some(0.5));
    assert_eq!(legend_tick_rel(&cmap, 20.0), Some(1.0));
}

#[test]
fn format_tick_matches_production_label_formatting() {
    assert_eq!(format_tick(500.0), "500");
    assert_eq!(format_tick(0.5), "0.5");
    assert_eq!(format_tick(-12.0), "-12");
    assert_eq!(format_tick(2.50), "2.5");
}
