use rustwx_core::{CanonicalField, FieldSelector, VerticalSelector};
use rustwx_models::{PlotRecipe, RenderStyle};
use rustwx_render::{
    Color, ColorScale, ContourLayer, ContourLinePattern, DiscreteColorScale, DomainFrame,
    DomainFrameSource, ExtendMode, LegendControls, LegendMode, LevelDensity, MapRenderRequest,
    ProductVisualMode, RenderDensity, WindStreamlineStyle,
    weather::{
        WeatherPalette, dewpoint_palette_celsius_for_levels, temperature_palette_cropped_f,
        weather_palette, winds_palette_segments,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct StaticPlotDesign {
    pub bounds: (f64, f64, f64, f64),
    pub visual_mode: ProductVisualMode,
    pub overlay_only: bool,
}

impl StaticPlotDesign {
    pub fn new(bounds: (f64, f64, f64, f64), visual_mode: ProductVisualMode) -> Self {
        Self {
            bounds,
            visual_mode,
            overlay_only: false,
        }
    }

    pub fn overlay_only(mut self, overlay_only: bool) -> Self {
        self.overlay_only = overlay_only;
        self
    }

    pub fn apply_to_request(self, request: &mut MapRenderRequest) {
        apply_static_map_design(request, self.bounds, self.visual_mode, self.overlay_only);
    }
}

pub fn longitude_bounds_span_deg(bounds: (f64, f64, f64, f64)) -> f64 {
    let raw_span = (bounds.1 - bounds.0).abs();
    if raw_span >= 359.0 {
        return raw_span.min(360.0);
    }

    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    if west <= east {
        east - west
    } else {
        east + 360.0 - west
    }
}

pub fn is_global_scale_domain(bounds: (f64, f64, f64, f64)) -> bool {
    let lat_span = (bounds.3 - bounds.2).abs();
    lat_span >= 100.0 && longitude_bounds_span_deg(bounds) >= 300.0
}

pub fn static_domain_frame_for_bounds(bounds: (f64, f64, f64, f64)) -> Option<DomainFrame> {
    if is_global_scale_domain(bounds) {
        None
    } else if straight_western_domain_frame_enabled(bounds) {
        Some(static_map_viewport_domain_frame())
    } else {
        Some(static_model_data_domain_frame())
    }
}

fn static_map_viewport_domain_frame() -> DomainFrame {
    DomainFrame {
        inset_px: 2,
        outline_width: 2,
        source: DomainFrameSource::MapViewport,
        ..DomainFrame::map_viewport_default()
    }
}

fn static_model_data_domain_frame() -> DomainFrame {
    DomainFrame {
        inset_px: 2,
        outline_width: 2,
        source: DomainFrameSource::ProjectedGrid,
        ..DomainFrame::map_viewport_default()
    }
}

fn straight_western_domain_frame_enabled(bounds: (f64, f64, f64, f64)) -> bool {
    let default = is_straight_western_domain_frame_candidate(bounds);
    std::env::var("RUSTWX_STRAIGHT_WEST_PROJECTION")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "mercator" | "straight" | "northup"
            )
        })
        .unwrap_or(default)
}

fn is_straight_western_domain_frame_candidate(bounds: (f64, f64, f64, f64)) -> bool {
    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    if west > east {
        return false;
    }
    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_bounds_span_deg(bounds);
    bounds.2 >= 25.0
        && bounds.3 <= 55.0
        && west >= -130.0
        && west <= -115.0
        && east >= -123.0
        && east <= -104.0
        && lat_span >= 4.0
        && lon_span <= 28.0
}

pub fn apply_static_map_design(
    request: &mut MapRenderRequest,
    bounds: (f64, f64, f64, f64),
    visual_mode: ProductVisualMode,
    overlay_only: bool,
) {
    request.visual_mode = visual_mode;
    request.render_density = RenderDensity {
        fill: high_detail_fill_density(),
        palette_multiplier: 4,
    };
    request.legend = LegendControls {
        density: LevelDensity::default(),
        mode: LegendMode::SmoothRamp,
    };
    if is_global_scale_domain(bounds) && !overlay_only {
        request.render_density = RenderDensity::default();
        request.legend = LegendControls {
            density: LevelDensity::default(),
            mode: LegendMode::SmoothRamp,
        };
    }
    request.domain_frame = static_domain_frame_for_bounds(bounds);
}

fn high_detail_fill_density() -> LevelDensity {
    LevelDensity {
        multiplier: 4,
        min_source_level_count: 2,
    }
}

pub fn operational_fill_scale_for_recipe(
    recipe: &PlotRecipe,
    filled_selector: FieldSelector,
) -> ColorScale {
    if recipe.slug == "mslp_10m_winds" || recipe.slug == "gefs_avg_mslp_10m_winds" {
        return ColorScale::Discrete(ten_meter_wind_speed_scale());
    }

    if filled_selector.field == CanonicalField::SmokeMassDensity {
        return ColorScale::Discrete(DiscreteColorScale {
            levels: vec![10.0, 20.0, 35.0, 55.0, 100.0, 150.0, 250.0, 500.0],
            colors: smoke_scale_colors(),
            extend: ExtendMode::Max,
            mask_below: Some(10.0),
        });
    }
    if filled_selector.field == CanonicalField::ColumnIntegratedSmoke {
        return ColorScale::Discrete(DiscreteColorScale {
            levels: vec![20.0, 40.0, 80.0, 160.0, 320.0],
            colors: smoke_scale_colors(),
            extend: ExtendMode::Max,
            mask_below: Some(20.0),
        });
    }

    let discrete = match recipe.style {
        RenderStyle::WeatherTemperature => {
            let (lo, hi, step, crop_f) = match filled_selector.vertical {
                VerticalSelector::HeightAboveGroundMeters(2) => {
                    (-60.0, 120.0, 1.0, Some((-60.0, 120.0)))
                }
                VerticalSelector::IsobaricHpa(200) => (-70.0, -29.0, 1.0, Some((-40.0, 70.0))),
                VerticalSelector::IsobaricHpa(250) => (-70.0, -29.0, 1.0, Some((-40.0, 70.0))),
                VerticalSelector::IsobaricHpa(300) => (-70.0, -29.0, 1.0, Some((-40.0, 70.0))),
                VerticalSelector::IsobaricHpa(500) => (-50.0, 6.0, 1.0, Some((-40.0, 70.0))),
                VerticalSelector::IsobaricHpa(700) => (-40.0, 26.0, 1.0, Some((-40.0, 90.0))),
                VerticalSelector::IsobaricHpa(850) => (-40.0, 40.0, 5.0, Some((-40.0, 110.0))),
                _ => (-50.0, 50.5, 0.5, Some((-40.0, 120.0))),
            };
            DiscreteColorScale {
                levels: range_step(lo, hi, step),
                colors: temperature_palette_cropped_f(
                    crop_f,
                    (((hi - lo) / step).round() as usize).max(2),
                ),
                extend: ExtendMode::Both,
                mask_below: None,
            }
        }
        RenderStyle::WeatherReflectivity | RenderStyle::WeatherRadarReflectivity => {
            reflectivity_dbz_scale()
        }
        RenderStyle::WeatherRh => relative_humidity_scale_for_selector(filled_selector),
        RenderStyle::WeatherProbability => DiscreteColorScale {
            levels: range_step(0.0, 101.0, 1.0),
            colors: weather_palette(WeatherPalette::Rh),
            extend: ExtendMode::Both,
            mask_below: None,
        },
        RenderStyle::WeatherVorticity => DiscreteColorScale {
            levels: range_step(-40.0, 60.1, 1.0),
            colors: weather_palette(WeatherPalette::RelVort),
            extend: ExtendMode::Both,
            mask_below: None,
        },
        RenderStyle::WeatherDewpoint => dewpoint_scale_for_selector(filled_selector),
        RenderStyle::WeatherPressure => mslp_pressure_fill_scale(),
        RenderStyle::WeatherHeight => DiscreteColorScale {
            levels: match filled_selector.vertical {
                VerticalSelector::IsobaricHpa(200) | VerticalSelector::IsobaricHpa(250) => {
                    range_step(50.0, 170.0, 5.0)
                }
                VerticalSelector::IsobaricHpa(300) => range_step(20.0, 160.0, 5.0),
                VerticalSelector::IsobaricHpa(500) => range_step(20.0, 140.0, 5.0),
                VerticalSelector::IsobaricHpa(700) => range_step(20.0, 80.0, 5.0),
                VerticalSelector::IsobaricHpa(850) | VerticalSelector::IsobaricHpa(925) => {
                    range_step(20.0, 80.0, 5.0)
                }
                _ => range_step(10.0, 71.0, 1.0),
            },
            colors: match filled_selector.vertical {
                VerticalSelector::IsobaricHpa(200) | VerticalSelector::IsobaricHpa(250) => {
                    winds_palette_segments(120)
                }
                VerticalSelector::IsobaricHpa(300) => winds_palette_segments(100),
                VerticalSelector::IsobaricHpa(500) => winds_palette_segments(100),
                VerticalSelector::IsobaricHpa(700)
                | VerticalSelector::IsobaricHpa(850)
                | VerticalSelector::IsobaricHpa(925) => winds_palette_segments(70),
                _ => winds_palette_segments(60),
            },
            extend: ExtendMode::Both,
            mask_below: Some(match filled_selector.vertical {
                VerticalSelector::IsobaricHpa(200) | VerticalSelector::IsobaricHpa(250) => 50.0,
                VerticalSelector::IsobaricHpa(300)
                | VerticalSelector::IsobaricHpa(500)
                | VerticalSelector::IsobaricHpa(700)
                | VerticalSelector::IsobaricHpa(850)
                | VerticalSelector::IsobaricHpa(925) => 20.0,
                _ => 10.0,
            }),
        },
        RenderStyle::WeatherWindGust | RenderStyle::WeatherWinds => {
            wind_speed_scale_for_selector(filled_selector)
        }
        RenderStyle::WeatherUh => DiscreteColorScale {
            levels: {
                let mut levels = range_step(0.0, 200.0, 5.0);
                levels.extend(range_step(200.0, 401.0, 10.0).into_iter().skip(1));
                levels
            },
            colors: weather_palette(WeatherPalette::Uh),
            extend: ExtendMode::Both,
            mask_below: Some(0.0),
        },
        RenderStyle::WeatherCloudCover => cloud_cover_scale(),
        RenderStyle::WeatherPrecipitableWater => precipitable_water_inches_scale(),
        RenderStyle::WeatherQpf => crate::qpf::qpf_inches_scale(),
        RenderStyle::WeatherCategorical => DiscreteColorScale {
            levels: vec![0.0, 0.5, 1.0],
            colors: vec![
                Color::rgba(242, 242, 242, 255),
                Color::rgba(216, 34, 34, 255),
            ],
            extend: ExtendMode::Neither,
            mask_below: Some(0.5),
        },
        RenderStyle::WeatherVisibility => DiscreteColorScale {
            levels: range_step(0.0, 10.5, 0.5),
            colors: weather_palette(WeatherPalette::MlMetric),
            extend: ExtendMode::Both,
            mask_below: None,
        },
        RenderStyle::WeatherSatellite => DiscreteColorScale {
            levels: range_step(170.0, 321.0, 2.0),
            colors: weather_palette(WeatherPalette::SimIr),
            extend: ExtendMode::Both,
            mask_below: None,
        },
        RenderStyle::WeatherLightning => DiscreteColorScale {
            levels: range_step(0.0, 20.5, 0.5),
            colors: weather_palette(WeatherPalette::Uh),
            extend: ExtendMode::Max,
            mask_below: Some(0.5),
        },
        _ => DiscreteColorScale {
            levels: range_step(-50.0, 5.0, 1.0),
            colors: weather_palette(WeatherPalette::Temperature),
            extend: ExtendMode::Both,
            mask_below: None,
        },
    };
    ColorScale::Discrete(discrete)
}

pub fn operational_contour_layer_for_values(
    selector: FieldSelector,
    values: &[f32],
) -> Option<ContourLayer> {
    let data = if selector.field == CanonicalField::GeopotentialHeight {
        values.iter().map(|value| value * 0.1).collect()
    } else if selector.field == CanonicalField::PressureReducedToMeanSeaLevel {
        values.iter().map(|value| value * 0.01).collect()
    } else {
        values.to_vec()
    };
    let (levels, color, width, labels, major_every, major_width, show_extrema) = match selector {
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(200),
            ..
        } => operational_height_contour_policy(range_step(1020.0, 1321.0, 6.0)),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(300),
            ..
        } => operational_height_contour_policy(range_step(700.0, 1101.0, 6.0)),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(250),
            ..
        } => operational_height_contour_policy(range_step(900.0, 1201.0, 6.0)),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(500),
            ..
        } => operational_height_contour_policy(range_step(450.0, 651.0, 6.0)),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(700),
            ..
        } => operational_height_contour_policy(range_step(100.0, 401.0, 6.0)),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(850),
            ..
        } => operational_height_contour_policy(range_step(0.0, 201.0, 6.0)),
        FieldSelector {
            field: CanonicalField::PressureReducedToMeanSeaLevel,
            ..
        } => operational_pressure_contour_policy(range_step(960.0, 1045.0, 2.0)),
        FieldSelector {
            field: CanonicalField::UpdraftHelicity,
            vertical:
                rustwx_core::VerticalSelector::HeightAboveGroundLayerMeters {
                    bottom_m: 2000,
                    top_m: 5000,
                },
            ..
        } => (vec![75.0], Color::BLACK, 1, false, None, None, false),
        _ => (
            range_step(0.0, 200.0, 10.0),
            Color::BLACK,
            1,
            true,
            Some(2),
            Some(2),
            false,
        ),
    };

    Some(ContourLayer {
        data,
        levels,
        color,
        width,
        labels,
        show_extrema,
        pattern: ContourLinePattern::Solid,
        major_every,
        major_width,
    })
}

pub fn operational_wind_streamline_style(stride_x: usize, stride_y: usize) -> WindStreamlineStyle {
    WindStreamlineStyle {
        stride_x: stride_x.max(1),
        stride_y: stride_y.max(1),
        color: Color::rgba(18, 24, 32, 72),
        width: 1,
        max_steps: 14,
        step_cells: 0.80,
        min_speed: 2.5,
    }
}

fn operational_height_contour_policy(
    levels: Vec<f64>,
) -> (Vec<f64>, Color, u32, bool, Option<usize>, Option<u32>, bool) {
    (
        levels,
        Color::rgba(0, 0, 0, 220),
        1,
        true,
        Some(2),
        Some(2),
        false,
    )
}

fn operational_pressure_contour_policy(
    levels: Vec<f64>,
) -> (Vec<f64>, Color, u32, bool, Option<usize>, Option<u32>, bool) {
    (levels, Color::BLACK, 1, true, Some(2), Some(2), true)
}

/// The catalog fill scale for MSLP values in hPa (the `WeatherPressure`
/// recipe arm, e.g. the HREF MSLP mean fill). NOTE: the production
/// `mslp_10m_winds` plot fills the companion 10 m wind speed and only
/// CONTOURS mslp at 2 hPa — no single-model production colorbar shows
/// these pressure values, which is why the store viewer resolver keeps the
/// generic ramp for the stored `mslp` plane instead of this scale.
pub(crate) fn mslp_pressure_fill_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: range_step(960.0, 1045.0, 2.0),
        colors: weather_palette(WeatherPalette::Winds),
        extend: ExtendMode::Both,
        mask_below: None,
    }
}

fn reflectivity_dbz_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: vec![
            10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0,
        ],
        colors: vec![
            Color::rgba(242, 246, 252, 255),
            Color::rgba(150, 183, 232, 255),
            Color::rgba(55, 105, 195, 255),
            Color::rgba(20, 94, 133, 255),
            Color::rgba(45, 126, 76, 255),
            Color::rgba(132, 169, 80, 255),
            Color::rgba(246, 226, 82, 255),
            Color::rgba(237, 143, 42, 255),
            Color::rgba(211, 32, 28, 255),
            Color::rgba(147, 5, 21, 255),
            Color::rgba(132, 34, 157, 255),
            Color::rgba(178, 178, 178, 255),
        ],
        extend: ExtendMode::Max,
        mask_below: Some(10.0),
    }
}

fn wind_speed_scale_for_selector(selector: FieldSelector) -> DiscreteColorScale {
    let levels = match selector.vertical {
        VerticalSelector::IsobaricHpa(200) | VerticalSelector::IsobaricHpa(250) => {
            range_step(50.0, 170.0, 5.0)
        }
        VerticalSelector::IsobaricHpa(500) => range_step(20.0, 140.0, 5.0),
        VerticalSelector::IsobaricHpa(700)
        | VerticalSelector::IsobaricHpa(850)
        | VerticalSelector::IsobaricHpa(925) => range_step(20.0, 80.0, 5.0),
        VerticalSelector::HeightAboveGroundMeters(10) => range_step(10.0, 60.0, 5.0),
        _ => range_step(10.0, 80.0, 5.0),
    };
    DiscreteColorScale {
        levels,
        colors: winds_palette_segments(90),
        extend: ExtendMode::Max,
        mask_below: Some(match selector.vertical {
            VerticalSelector::IsobaricHpa(200) | VerticalSelector::IsobaricHpa(250) => 50.0,
            VerticalSelector::IsobaricHpa(500)
            | VerticalSelector::IsobaricHpa(700)
            | VerticalSelector::IsobaricHpa(850)
            | VerticalSelector::IsobaricHpa(925) => 20.0,
            VerticalSelector::HeightAboveGroundMeters(10) => 10.0,
            _ => 10.0,
        }),
    }
}

fn ten_meter_wind_speed_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: range_step(10.0, 60.0, 5.0),
        colors: winds_palette_segments(60),
        extend: ExtendMode::Max,
        mask_below: Some(10.0),
    }
}

fn dewpoint_scale_for_selector(selector: FieldSelector) -> DiscreteColorScale {
    match selector.vertical {
        VerticalSelector::HeightAboveGroundMeters(2) => {
            let levels = range_step(-40.0, 90.0, 1.0);
            DiscreteColorScale {
                colors: surface_dewpoint_colors(),
                levels,
                extend: ExtendMode::Both,
                mask_below: None,
            }
        }
        VerticalSelector::IsobaricHpa(_) => {
            let levels = range_step(-40.0, 31.0, 1.0);
            DiscreteColorScale {
                colors: dewpoint_palette_celsius_for_levels(&levels),
                levels,
                extend: ExtendMode::Both,
                mask_below: None,
            }
        }
        _ => {
            let levels = range_step(-40.0, 90.0, 1.0);
            DiscreteColorScale {
                colors: surface_dewpoint_colors(),
                levels,
                extend: ExtendMode::Both,
                mask_below: None,
            }
        }
    }
}

fn relative_humidity_scale_for_selector(selector: FieldSelector) -> DiscreteColorScale {
    match selector.vertical {
        VerticalSelector::HeightAboveGroundMeters(2) => DiscreteColorScale {
            levels: range_step(0.0, 100.0, 5.0),
            colors: surface_relative_humidity_colors(),
            extend: ExtendMode::Max,
            mask_below: None,
        },
        _ => DiscreteColorScale {
            levels: range_step(0.0, 101.0, 1.0),
            colors: weather_palette(WeatherPalette::Rh),
            extend: ExtendMode::Both,
            mask_below: None,
        },
    }
}

fn surface_dewpoint_colors() -> Vec<Color> {
    let mut colors = weather_palette(WeatherPalette::Dewpoint);
    if colors.len() <= 1 {
        return colors;
    }

    colors.remove(0);
    if let Some(last) = colors.last().copied() {
        colors.push(last);
    }
    colors
}

fn surface_relative_humidity_colors() -> Vec<Color> {
    vec![
        Color::rgba(140, 45, 4, 255),
        Color::rgba(204, 76, 2, 255),
        Color::rgba(236, 112, 20, 255),
        Color::rgba(254, 153, 41, 255),
        Color::rgba(254, 196, 79, 255),
        Color::rgba(255, 247, 188, 255),
        Color::rgba(224, 243, 219, 255),
        Color::rgba(168, 221, 181, 255),
        Color::rgba(67, 162, 202, 255),
        Color::rgba(8, 104, 172, 255),
    ]
}

fn cloud_cover_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: range_step(10.0, 100.0, 10.0),
        colors: vec![
            Color::rgba(255, 255, 255, 255),
            Color::rgba(222, 222, 222, 255),
            Color::rgba(178, 178, 178, 255),
            Color::rgba(128, 128, 128, 255),
            Color::rgba(70, 80, 100, 255),
            Color::rgba(35, 68, 122, 255),
            Color::rgba(38, 111, 166, 255),
            Color::rgba(103, 177, 209, 255),
            Color::rgba(189, 232, 241, 255),
        ],
        extend: ExtendMode::Both,
        mask_below: None,
    }
}

fn precipitable_water_inches_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: vec![
            0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00,
        ],
        colors: vec![
            Color::rgba(70, 55, 44, 255),
            Color::rgba(118, 108, 94, 255),
            Color::rgba(213, 211, 189, 255),
            Color::rgba(183, 224, 175, 255),
            Color::rgba(105, 191, 105, 255),
            Color::rgba(32, 137, 67, 255),
            Color::rgba(16, 111, 101, 255),
            Color::rgba(39, 124, 158, 255),
            Color::rgba(63, 95, 168, 255),
            Color::rgba(116, 74, 165, 255),
            Color::rgba(191, 127, 177, 255),
        ],
        extend: ExtendMode::Both,
        mask_below: None,
    }
}

fn smoke_scale_colors() -> Vec<Color> {
    vec![
        Color::rgba(82, 185, 226, 42),
        Color::rgba(84, 210, 238, 78),
        Color::rgba(116, 230, 140, 116),
        Color::rgba(247, 232, 65, 160),
        Color::rgba(255, 169, 42, 200),
        Color::rgba(244, 76, 31, 226),
        Color::rgba(218, 10, 36, 238),
        Color::rgba(135, 0, 150, 246),
        Color::rgba(78, 0, 138, 252),
        Color::rgba(48, 0, 112, 255),
    ]
}

fn range_step(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut value = start;
    while value <= stop + 1e-9 {
        out.push(value);
        value += step;
    }
    out
}

fn normalize_longitude_for_bounds(lon: f64) -> f64 {
    let mut lon = lon % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_core::{CanonicalField, Field2D, FieldSelector, GridShape, LatLonGrid, ProductKey};
    use rustwx_render::{ColorScale, DiscreteColorScale, ExtendMode};

    fn sample_request() -> MapRenderRequest {
        let shape = GridShape::new(2, 2).unwrap();
        let grid = LatLonGrid::new(
            shape,
            vec![35.0, 35.0, 36.0, 36.0],
            vec![-100.0, -99.0, -100.0, -99.0],
        )
        .unwrap();
        let field = Field2D::new(
            ProductKey::named("sample"),
            "unit",
            grid,
            vec![0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        MapRenderRequest::new(
            field.into(),
            ColorScale::Discrete(DiscreteColorScale {
                levels: vec![0.0, 1.0, 2.0, 3.0],
                colors: vec![
                    rustwx_render::Color::rgba(0, 0, 255, 255),
                    rustwx_render::Color::rgba(255, 0, 0, 255),
                ],
                extend: ExtendMode::Neither,
                mask_below: None,
            }),
        )
    }

    #[test]
    fn regional_static_design_uses_projected_grid_frame_and_smooth_legend() {
        let mut request = sample_request();

        StaticPlotDesign::new(
            (-125.0, -66.0, 24.0, 50.0),
            ProductVisualMode::FilledMeteorology,
        )
        .apply_to_request(&mut request);

        assert_eq!(request.visual_mode, ProductVisualMode::FilledMeteorology);
        assert_eq!(
            request.domain_frame.map(|frame| frame.source),
            Some(DomainFrameSource::ProjectedGrid)
        );
        assert_eq!(request.legend.mode, LegendMode::SmoothRamp);
        assert_eq!(request.render_density.fill, high_detail_fill_density());
        assert_eq!(request.render_density.palette_multiplier, 4);
    }

    #[test]
    fn straight_west_static_design_uses_viewport_frame() {
        let mut request = sample_request();

        StaticPlotDesign::new(
            (-124.9, -113.8, 31.9, 42.5),
            ProductVisualMode::FilledMeteorology,
        )
        .apply_to_request(&mut request);

        assert_eq!(
            request.domain_frame.map(|frame| frame.source),
            Some(DomainFrameSource::MapViewport)
        );

        let mut west_request = sample_request();
        StaticPlotDesign::new(
            (-125.7, -110.5, 30.5, 49.0),
            ProductVisualMode::FilledMeteorology,
        )
        .apply_to_request(&mut west_request);
        assert_eq!(
            west_request.domain_frame.map(|frame| frame.source),
            Some(DomainFrameSource::MapViewport)
        );
    }

    #[test]
    fn rockies_static_design_keeps_projected_grid_frame() {
        let mut request = sample_request();

        StaticPlotDesign::new(
            (-112.0, -96.0, 37.0, 49.5),
            ProductVisualMode::FilledMeteorology,
        )
        .apply_to_request(&mut request);

        assert_eq!(
            request.domain_frame.map(|frame| frame.source),
            Some(DomainFrameSource::ProjectedGrid)
        );
    }

    #[test]
    fn global_filled_static_design_uses_smooth_legend_without_viewport_frame() {
        let mut request = sample_request();

        apply_static_map_design(
            &mut request,
            (-180.0, 179.999, -90.0, 90.0),
            ProductVisualMode::FilledMeteorology,
            false,
        );

        assert!(request.domain_frame.is_none());
        assert_eq!(request.legend.mode, LegendMode::SmoothRamp);
        assert_eq!(request.render_density, RenderDensity::default());
    }

    #[test]
    fn global_overlay_static_design_keeps_stepped_legend_policy() {
        let mut request = sample_request();

        apply_static_map_design(
            &mut request,
            (-180.0, 179.999, -90.0, 90.0),
            ProductVisualMode::OverlayAnalysis,
            true,
        );

        assert!(request.domain_frame.is_none());
        assert_eq!(request.legend.mode, LegendMode::SmoothRamp);
        assert_eq!(request.render_density.fill, high_detail_fill_density());
        assert_eq!(request.render_density.palette_multiplier, 4);
    }

    #[test]
    fn operational_pressure_contours_convert_units_and_mark_extrema() {
        let layer = operational_contour_layer_for_values(
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
            &[100000.0, 100200.0, 100400.0, 100600.0],
        )
        .expect("pressure contour layer");

        assert_eq!(layer.data[0], 1000.0);
        assert_eq!(layer.levels.first().copied(), Some(960.0));
        assert_eq!(layer.width, 1);
        assert_eq!(layer.major_every, Some(2));
        assert_eq!(layer.major_width, Some(2));
        assert_eq!(layer.pattern, ContourLinePattern::Solid);
        assert!(layer.labels);
        assert!(layer.show_extrema);
    }

    #[test]
    fn operational_height_contours_convert_to_decameters_without_extrema() {
        let layer = operational_contour_layer_for_values(
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500),
            &[5400.0, 5460.0, 5520.0, 5580.0],
        )
        .expect("height contour layer");

        assert_eq!(layer.data[0], 540.0);
        assert_eq!(layer.levels.first().copied(), Some(450.0));
        assert_eq!(layer.levels.get(1).copied(), Some(456.0));
        assert_eq!(layer.color, Color::rgba(0, 0, 0, 220));
        assert_eq!(layer.major_every, Some(2));
        assert_eq!(layer.major_width, Some(2));
        assert!(layer.labels);
        assert!(!layer.show_extrema);
    }

    #[test]
    fn operational_wind_streamlines_are_subtle_dense_flow_texture() {
        let style = operational_wind_streamline_style(9, 7);

        assert_eq!(style.stride_x, 9);
        assert_eq!(style.stride_y, 7);
        assert_eq!(style.width, 1);
        assert!(style.color.a < 160);
        assert!(style.max_steps >= 12);
        assert!(style.step_cells > 0.0);
    }

    #[test]
    fn operational_fill_scale_masks_sparse_signal_products() {
        let reflectivity = rustwx_models::plot_recipe("composite_reflectivity").unwrap();
        let ColorScale::Discrete(reflectivity_scale) = operational_fill_scale_for_recipe(
            reflectivity,
            FieldSelector::surface(CanonicalField::CompositeReflectivity),
        ) else {
            panic!("expected reflectivity discrete scale");
        };
        assert_eq!(reflectivity_scale.levels.first().copied(), Some(10.0));
        assert_eq!(reflectivity_scale.levels.last().copied(), Some(70.0));
        assert_eq!(reflectivity_scale.extend, ExtendMode::Max);
        assert_eq!(reflectivity_scale.mask_below, Some(10.0));

        let mslp_winds = rustwx_models::plot_recipe("mslp_10m_winds").unwrap();
        let ColorScale::Discrete(mslp_wind_scale) = operational_fill_scale_for_recipe(
            mslp_winds,
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
        ) else {
            panic!("expected MSLP/10m wind discrete scale");
        };
        assert_eq!(mslp_wind_scale.levels.first().copied(), Some(10.0));
        assert_eq!(mslp_wind_scale.mask_below, Some(10.0));

        let qpf = rustwx_models::plot_recipe("1h_qpf").unwrap();
        let ColorScale::Discrete(qpf_scale) = operational_fill_scale_for_recipe(
            qpf,
            FieldSelector::surface(CanonicalField::TotalPrecipitation),
        ) else {
            panic!("expected QPF discrete scale");
        };
        assert_eq!(qpf_scale.mask_below, Some(0.01));

        let categorical = rustwx_models::plot_recipe("categorical_snow").unwrap();
        let ColorScale::Discrete(categorical_scale) = operational_fill_scale_for_recipe(
            categorical,
            FieldSelector::surface(CanonicalField::CategoricalSnow),
        ) else {
            panic!("expected categorical discrete scale");
        };
        assert_eq!(categorical_scale.extend, ExtendMode::Neither);
        assert_eq!(categorical_scale.mask_below, Some(0.5));

        let surface_smoke = rustwx_models::plot_recipe("smoke_pm25_native").unwrap();
        let ColorScale::Discrete(surface_smoke_scale) = operational_fill_scale_for_recipe(
            surface_smoke,
            FieldSelector::height_agl(CanonicalField::SmokeMassDensity, 8),
        ) else {
            panic!("expected surface smoke discrete scale");
        };
        assert_eq!(surface_smoke_scale.levels.first().copied(), Some(10.0));
        assert_eq!(surface_smoke_scale.mask_below, Some(10.0));
        assert!(surface_smoke_scale.colors[0].a < 80);

        let column_smoke = rustwx_models::plot_recipe("smoke_column").unwrap();
        let ColorScale::Discrete(column_smoke_scale) = operational_fill_scale_for_recipe(
            column_smoke,
            FieldSelector::entire_atmosphere(CanonicalField::ColumnIntegratedSmoke),
        ) else {
            panic!("expected column smoke discrete scale");
        };
        assert_eq!(column_smoke_scale.levels.first().copied(), Some(20.0));
        assert_eq!(column_smoke_scale.mask_below, Some(20.0));
        assert!(column_smoke_scale.colors[0].a < 80);
    }
}
