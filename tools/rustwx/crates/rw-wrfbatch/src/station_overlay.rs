//! Scalar station marks shared by single maps and comparison sheets.

use rustwx_products::geographic_overlays::{MapOverlays, ValuePointLayer, ValuePointSpec};
use rustwx_render::{ColormapBuildOptions, RgbaImage, StaticPlotStyle, build_colormap};

use crate::compare::{difference_scale, difference_step};

/// Observed and sampled forecast values in the map's display units.
#[derive(Debug, Clone)]
pub struct StationDot {
    pub latitude: f64,
    pub longitude: f64,
    pub observed: f64,
    pub forecast: Option<f64>,
}

/// Observed dots wear the exact field palette. Error dots wear a fixed,
/// symmetric observed-minus-forecast ladder in the same display units.
pub fn scalar_dots(
    dots: &[StationDot],
    display_units: &str,
    error: bool,
) -> Result<MapOverlays, String> {
    let scale = if error {
        let step = difference_step(display_units)
            .ok_or_else(|| format!("no fixed station error ladder for units {display_units:?}"))?;
        Some(difference_scale(step))
    } else {
        None
    };
    let points = dots
        .iter()
        .filter_map(|dot| {
            if !dot.latitude.is_finite() || !dot.longitude.is_finite() || !dot.observed.is_finite()
            {
                return None;
            }
            let value = if error {
                dot.observed - dot.forecast.filter(|value| value.is_finite())?
            } else {
                dot.observed
            };
            Some(ValuePointSpec {
                lat: dot.latitude,
                lon: dot.longitude,
                value,
            })
        })
        .collect();
    Ok(MapOverlays {
        value_layers: vec![ValuePointLayer {
            products: Vec::new(),
            units: display_units.to_string(),
            points,
            scale,
            radius_px: 5,
        }],
        ..MapOverlays::default()
    })
}

/// Add a labelled station error key without resampling a map pixel.
pub fn append_error_key(
    sheet: &RgbaImage,
    display_units: &str,
    mut options: ColormapBuildOptions,
) -> Result<RgbaImage, String> {
    let step = difference_step(display_units)
        .ok_or_else(|| format!("no station error ladder for {display_units:?}"))?;
    let scale = difference_scale(step);
    let discrete = scale.resolved_discrete();
    options.render_density = StaticPlotStyle::from_env().render_density(options.render_density);
    let cmap = build_colormap(&scale, options);
    let background = *sheet.get_pixel(0, 0);
    let mut result = RgbaImage::from_pixel(sheet.width(), sheet.height() + 78, background);
    image::imageops::overlay(&mut result, sheet, 0, 0);
    let luminance = 0.2126 * f64::from(background[0])
        + 0.7152 * f64::from(background[1])
        + 0.0722 * f64::from(background[2]);
    let ink = if luminance >= 128.0 {
        rustwx_render::Rgba::BLACK
    } else {
        rustwx_render::Rgba::WHITE
    };
    let top = sheet.height() + 10;
    rustwx_render::draw_text(
        &mut result,
        &format!("Station observed minus forecast ({display_units})"),
        18,
        top as i32,
        ink,
        1,
    );
    let left = 18_u32;
    let usable = sheet.width().saturating_sub(36);
    let bins = discrete.colors.len() as u32;
    for bin in 0..bins {
        let x0 = left + usable * bin / bins;
        let x1 = left + usable * (bin + 1) / bins;
        let value = (discrete.levels[bin as usize] + discrete.levels[bin as usize + 1]) / 2.0;
        let color = cmap.map(value).to_image_rgba();
        for y in top + 22..top + 39 {
            for x in x0..x1.saturating_sub(1) {
                result.put_pixel(x, y, color);
            }
        }
        let label = if bin == 0 {
            format!("< {:+.1}", discrete.levels[1])
        } else if bin + 1 == bins {
            format!(">= {:+.1}", discrete.levels[bin as usize])
        } else {
            format!("{value:+.1}")
        };
        rustwx_render::draw_text(&mut result, &label, x0 as i32, (top + 45) as i32, ink, 1);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_forecasts_do_not_become_zero_error_dots() {
        let dots = vec![
            StationDot {
                latitude: 35.0,
                longitude: -98.0,
                observed: 70.0,
                forecast: Some(68.0),
            },
            StationDot {
                latitude: 35.5,
                longitude: -98.0,
                observed: 75.0,
                forecast: None,
            },
        ];
        let observed = scalar_dots(&dots, "degF", false).unwrap();
        assert_eq!(observed.value_layers[0].points.len(), 2);
        assert!(observed.value_layers[0].scale.is_none());
        let errors = scalar_dots(&dots, "degF", true).unwrap();
        assert_eq!(errors.value_layers[0].points.len(), 1);
        assert_eq!(errors.value_layers[0].points[0].value, 2.0);
        assert!(errors.value_layers[0].scale.is_some());
    }
}
