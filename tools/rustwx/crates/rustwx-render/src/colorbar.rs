use crate::color::Rgba;
use crate::colormap::{LegendMode, LeveledColormap};
use crate::presentation::ColorbarPresentation;
use image::RgbaImage;

/// Thin cool-gray frame for the colorbar — reads as a subtle divider rather
/// than a hard black rule. Matches the "modern" look where the colorbar's
/// color swatches are the main signal and chrome recedes.
const COLORBAR_FRAME: Rgba = Rgba {
    r: 90,
    g: 96,
    b: 108,
    a: 255,
};

/// Separator between adjacent color swatches. Dark enough to keep dense bars
/// visibly discrete instead of reading like a smooth gradient.
const COLORBAR_DIVIDER: Rgba = Rgba {
    r: 36,
    g: 40,
    b: 48,
    a: 190,
};

/// Draw a horizontal colorbar onto an existing image.
///
/// Fills the rectangle `(x, y, x+width, y+height)` with colour swatches
/// matching the levels in the colormap. Stepped intervals are sized by the
/// numeric level span so nonlinear scales keep their colors aligned with tick
/// labels.
pub fn draw_colorbar(
    img: &mut RgbaImage,
    cmap: &LeveledColormap,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    mode: LegendMode,
    presentation: ColorbarPresentation,
) {
    let legend_levels = cmap.legend_levels_for_display();
    let legend_colors = cmap.legend_colors_for_display();

    let n_intervals = if legend_levels.len() > 1 {
        legend_levels.len() - 1
    } else {
        return;
    };

    for px in x..x.saturating_add(width).min(img.width()) {
        let rel = ((px - x) as f64 + 0.5) / width.max(1) as f64;
        let color = match mode {
            LegendMode::Stepped => stepped_color_at_rel(legend_levels, legend_colors, rel),
            LegendMode::SmoothRamp => smooth_color_at_rel(legend_levels, legend_colors, rel),
        };
        for py in y..y.saturating_add(height).min(img.height()) {
            img.put_pixel(px, py, color.to_image_rgba());
        }
    }

    let x_end = (x + width).min(img.width());
    let y_end = (y + height).min(img.height());

    // Hairline separators between swatches — light, partial alpha so they
    // only suggest boundaries instead of chopping the bar into stripes.
    let divider_color = if presentation.divider_color == Rgba::TRANSPARENT {
        COLORBAR_DIVIDER
    } else {
        presentation.divider_color
    };
    if matches!(mode, LegendMode::Stepped) {
        for i in 1..n_intervals {
            let Some(frac) = level_fraction(legend_levels, legend_levels[i]) else {
                continue;
            };
            let tick_x = x + (frac * width as f64).round() as u32;
            if tick_x < img.width() {
                for py in (y + 1)..y_end.saturating_sub(1) {
                    // Alpha-composite onto the existing swatch so dense bars keep
                    // visible bin edges without turning into a full black grid.
                    let dst = img.get_pixel(tick_x, py).0;
                    let a = divider_color.a as f64 / 255.0;
                    let inv = 1.0 - a;
                    let blended = image::Rgba([
                        (divider_color.r as f64 * a + dst[0] as f64 * inv).round() as u8,
                        (divider_color.g as f64 * a + dst[1] as f64 * inv).round() as u8,
                        (divider_color.b as f64 * a + dst[2] as f64 * inv).round() as u8,
                        255,
                    ]);
                    img.put_pixel(tick_x, py, blended);
                }
            }
        }
    }

    // Thin cool-gray outer frame — one pixel, muted slate instead of solid black.
    let frame = if presentation.frame_color == Rgba::TRANSPARENT {
        COLORBAR_FRAME
    } else {
        presentation.frame_color
    }
    .to_image_rgba();
    for px in x..x_end {
        img.put_pixel(px, y, frame);
        if y_end > 0 {
            img.put_pixel(px, y_end - 1, frame);
        }
    }
    for py in y..y_end {
        img.put_pixel(x, py, frame);
        if x_end > 0 {
            img.put_pixel(x_end - 1, py, frame);
        }
    }
}

/// Draw a vertical colorbar, with low values at the bottom and high values at
/// the top.
pub fn draw_vertical_colorbar(
    img: &mut RgbaImage,
    cmap: &LeveledColormap,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    mode: LegendMode,
    presentation: ColorbarPresentation,
) {
    let legend_levels = cmap.legend_levels_for_display();
    let legend_colors = cmap.legend_colors_for_display();

    let n_intervals = if legend_levels.len() > 1 {
        legend_levels.len() - 1
    } else {
        return;
    };

    let x_end = x.saturating_add(width).min(img.width());
    let y_end = y.saturating_add(height).min(img.height());
    for py in y..y_end {
        let rel = 1.0 - ((py - y) as f64 + 0.5) / height.max(1) as f64;
        let color = match mode {
            LegendMode::Stepped => stepped_color_at_rel(legend_levels, legend_colors, rel),
            LegendMode::SmoothRamp => smooth_color_at_rel(legend_levels, legend_colors, rel),
        };
        for px in x..x_end {
            img.put_pixel(px, py, color.to_image_rgba());
        }
    }

    let divider_color = if presentation.divider_color == Rgba::TRANSPARENT {
        COLORBAR_DIVIDER
    } else {
        presentation.divider_color
    };
    if matches!(mode, LegendMode::Stepped) {
        for i in 1..n_intervals {
            let Some(frac) = level_fraction(legend_levels, legend_levels[i]) else {
                continue;
            };
            let tick_y = y + ((1.0 - frac) * height as f64).round() as u32;
            if tick_y < img.height() {
                for px in x.saturating_add(1)..x_end.saturating_sub(1) {
                    let dst = img.get_pixel(px, tick_y).0;
                    let a = divider_color.a as f64 / 255.0;
                    let inv = 1.0 - a;
                    let blended = image::Rgba([
                        (divider_color.r as f64 * a + dst[0] as f64 * inv).round() as u8,
                        (divider_color.g as f64 * a + dst[1] as f64 * inv).round() as u8,
                        (divider_color.b as f64 * a + dst[2] as f64 * inv).round() as u8,
                        255,
                    ]);
                    img.put_pixel(px, tick_y, blended);
                }
            }
        }
    }

    let frame = if presentation.frame_color == Rgba::TRANSPARENT {
        COLORBAR_FRAME
    } else {
        presentation.frame_color
    }
    .to_image_rgba();
    if y < img.height() {
        for px in x..x_end {
            img.put_pixel(px, y, frame);
        }
    }
    if y_end > 0 {
        let bottom = y_end - 1;
        for px in x..x_end {
            img.put_pixel(px, bottom, frame);
        }
    }
    if x < img.width() {
        for py in y..y_end {
            img.put_pixel(x, py, frame);
        }
    }
    if x_end > 0 {
        let right = x_end - 1;
        for py in y..y_end {
            img.put_pixel(right, py, frame);
        }
    }
}

/// The colorbar color at fractional position `rel` in `[0, 1]` (0 = the
/// lowest legend level, 1 = the highest) — exactly the per-pixel sampling
/// [`draw_colorbar`]/[`draw_vertical_colorbar`] paint, exposed so external
/// legend widgets reproduce the production colorbar's colors bit-for-bit.
pub fn legend_color_at_rel(cmap: &LeveledColormap, mode: LegendMode, rel: f64) -> Rgba {
    let legend_levels = cmap.legend_levels_for_display();
    let legend_colors = cmap.legend_colors_for_display();
    match mode {
        LegendMode::Stepped => stepped_color_at_rel(legend_levels, legend_colors, rel),
        LegendMode::SmoothRamp => smooth_color_at_rel(legend_levels, legend_colors, rel),
    }
}

/// Fractional position of `value` along the colorbar (0 = lowest legend
/// level, 1 = highest), the same linear-by-value placement the production
/// tick labels use. `None` when the legend level range is degenerate.
pub fn legend_tick_rel(cmap: &LeveledColormap, value: f64) -> Option<f64> {
    level_fraction(cmap.legend_levels_for_display(), value)
}

fn level_range(levels: &[f64]) -> Option<(f64, f64)> {
    let lo = *levels.first()?;
    let hi = *levels.last()?;
    (lo.is_finite() && hi.is_finite() && hi > lo).then_some((lo, hi))
}

fn value_at_rel(levels: &[f64], rel: f64) -> Option<f64> {
    let (lo, hi) = level_range(levels)?;
    Some(lo + rel.clamp(0.0, 1.0) * (hi - lo))
}

fn level_fraction(levels: &[f64], level: f64) -> Option<f64> {
    let (lo, hi) = level_range(levels)?;
    Some(((level - lo) / (hi - lo)).clamp(0.0, 1.0))
}

fn stepped_color_at_rel(levels: &[f64], colors: &[Rgba], rel: f64) -> Rgba {
    let Some(value) = value_at_rel(levels, rel) else {
        return Rgba::TRANSPARENT;
    };
    if colors.is_empty() || levels.len() < 2 {
        return Rgba::TRANSPARENT;
    }
    if value < levels[0] {
        return colors[0];
    }
    let n_intervals = levels.len() - 1;
    let idx = levels.partition_point(|level| *level <= value);
    colors[idx
        .saturating_sub(1)
        .min(n_intervals - 1)
        .min(colors.len() - 1)]
}

fn smooth_color_at_rel(levels: &[f64], colors: &[Rgba], rel: f64) -> Rgba {
    let Some(value) = value_at_rel(levels, rel) else {
        return Rgba::TRANSPARENT;
    };
    if colors.is_empty() || levels.len() < 2 {
        return Rgba::TRANSPARENT;
    }
    if colors.len() == 1 || value <= levels[0] {
        return colors[0];
    }
    if value >= levels[levels.len() - 1] {
        return colors[colors.len() - 1];
    }
    let interval = levels
        .partition_point(|level| *level <= value)
        .saturating_sub(1)
        .min(levels.len() - 2);
    let lo_level = levels[interval];
    let hi_level = levels[interval + 1];
    let lo_color = colors[interval.min(colors.len() - 1)];
    let hi_color = colors[(interval + 1).min(colors.len() - 1)];
    let t = if hi_level > lo_level {
        ((value - lo_level) / (hi_level - lo_level)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Rgba {
        r: (lo_color.r as f64 + (hi_color.r as f64 - lo_color.r as f64) * t).round() as u8,
        g: (lo_color.g as f64 + (hi_color.g as f64 - lo_color.g as f64) * t).round() as u8,
        b: (lo_color.b as f64 + (hi_color.b as f64 - lo_color.b as f64) * t).round() as u8,
        a: (lo_color.a as f64 + (hi_color.a as f64 - lo_color.a as f64) * t).round() as u8,
    }
}

/// Draw short tick marks at specified relative positions (0..1) hanging above
/// the colorbar. Callers own the label placement; this just draws the line.
pub fn draw_colorbar_ticks(
    img: &mut RgbaImage,
    cbar_x: u32,
    cbar_y: u32,
    cbar_width: u32,
    positions: &[f64],
    tick_color: Rgba,
) {
    let frame = if tick_color == Rgba::TRANSPARENT {
        COLORBAR_FRAME
    } else {
        tick_color
    }
    .to_image_rgba();
    if cbar_y < 4 {
        return;
    }
    for &frac in positions {
        if !(0.0..=1.0).contains(&frac) {
            continue;
        }
        let px = cbar_x + (frac * cbar_width as f64).round() as u32;
        if px >= img.width() {
            continue;
        }
        for dy in 1..=3 {
            let py = cbar_y.saturating_sub(dy);
            if py < img.height() {
                img.put_pixel(px, py, frame);
            }
        }
    }
}

pub fn draw_vertical_colorbar_ticks(
    img: &mut RgbaImage,
    cbar_x: u32,
    cbar_y: u32,
    cbar_width: u32,
    cbar_height: u32,
    positions: &[f64],
    tick_color: Rgba,
) {
    let frame = if tick_color == Rgba::TRANSPARENT {
        COLORBAR_FRAME
    } else {
        tick_color
    }
    .to_image_rgba();
    let tick_x = cbar_x.saturating_add(cbar_width);
    if tick_x >= img.width() {
        return;
    }
    for &frac in positions {
        if !(0.0..=1.0).contains(&frac) {
            continue;
        }
        let py = cbar_y + ((1.0 - frac) * cbar_height as f64).round() as u32;
        if py >= img.height() {
            continue;
        }
        for dx in 0..=3 {
            let px = tick_x.saturating_add(dx);
            if px < img.width() {
                img.put_pixel(px, py, frame);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colormap::{Extend, LeveledColormap};
    use crate::presentation::ColorbarOrientation;

    fn color(r: u8, g: u8, b: u8) -> Rgba {
        Rgba { r, g, b, a: 255 }
    }

    fn test_presentation() -> ColorbarPresentation {
        ColorbarPresentation {
            orientation: ColorbarOrientation::HorizontalBottom,
            frame_color: Rgba::TRANSPARENT,
            divider_color: Rgba::TRANSPARENT,
            tick_color: Rgba::TRANSPARENT,
            label_color: Rgba::TRANSPARENT,
        }
    }

    #[test]
    fn stepped_colorbar_uses_numeric_level_widths() {
        let white = color(255, 255, 255);
        let green = color(0, 180, 0);
        let blue = color(0, 0, 220);
        let pink = color(220, 0, 180);
        let cmap = LeveledColormap {
            levels: vec![0.0, 0.1, 0.5, 1.0, 2.0],
            colors: vec![white, green, blue, pink],
            legend_levels: vec![0.0, 0.1, 0.5, 1.0, 2.0],
            legend_colors: vec![white, green, blue, pink],
            under_color: None,
            over_color: None,
            mask_below: None,
        };
        let mut img = RgbaImage::new(200, 8);

        draw_colorbar(
            &mut img,
            &cmap,
            0,
            0,
            200,
            8,
            LegendMode::Stepped,
            test_presentation(),
        );

        assert_eq!(img.get_pixel(4, 4).0, white.to_image_rgba().0);
        assert_eq!(img.get_pixel(15, 4).0, green.to_image_rgba().0);
        assert_eq!(img.get_pixel(55, 4).0, blue.to_image_rgba().0);
        assert_eq!(img.get_pixel(110, 4).0, pink.to_image_rgba().0);
    }

    #[test]
    fn vertical_colorbar_places_high_values_at_top() {
        let blue = color(0, 0, 220);
        let red = color(220, 0, 0);
        let cmap = LeveledColormap {
            levels: vec![0.0, 10.0, 20.0],
            colors: vec![blue, red],
            legend_levels: vec![0.0, 10.0, 20.0],
            legend_colors: vec![blue, red],
            under_color: None,
            over_color: None,
            mask_below: None,
        };
        let mut img = RgbaImage::new(12, 120);

        draw_vertical_colorbar(
            &mut img,
            &cmap,
            0,
            0,
            8,
            120,
            LegendMode::Stepped,
            test_presentation(),
        );

        assert_eq!(img.get_pixel(4, 4).0, red.to_image_rgba().0);
        assert_eq!(img.get_pixel(4, 114).0, blue.to_image_rgba().0);
    }

    #[test]
    fn listed_palette_qpf_breakpoints_land_on_expected_families() {
        let levels = vec![0.0, 0.1, 0.5, 1.0, 15.0];
        let palette = crate::weather::weather_palette(crate::weather::WeatherPalette::Precip)
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>();
        let cmap = LeveledColormap::from_palette(&palette, &levels, Extend::Max, None);

        let trace = cmap.map(0.05);
        let green = cmap.map(0.11);
        let blue = cmap.map(0.51);
        let pink = cmap.map(1.01);

        assert!(green.g > green.r && green.g >= green.b);
        assert!(blue.b > blue.r && blue.b >= blue.g);
        assert!(pink.r > trace.r && pink.b > trace.b);
    }
}
