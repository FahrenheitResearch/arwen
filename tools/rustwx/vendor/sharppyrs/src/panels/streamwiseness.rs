//! "Streamwiseness" inset — port of `sharpmod.viz.streamwiseness`
//! (`streamwiseness_profile` + `plotStreamwiseness`): the fraction of
//! horizontal vorticity aligned with the storm-relative wind through 6 km AGL,
//! shaded cyclonic (red) / anticyclonic (blue), with dashed threshold markers
//! at 0.5 / 1 / 3 km.

use egui::epaint::TextShape;
use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Shape, Stroke, StrokeKind, Vec2};

use crate::Profile;
use crate::derived::DerivedParams;
use crate::skewt::SkewTStyle;
use crate::utils::qc;

const KTS_TO_MS: f64 = 0.5144444444444445;
const MAX_HEIGHT_KM: f64 = 6.0;
const RIGHT_INSET: f32 = 25.0;

const PROFILE_COLOR: Color32 = Color32::from_rgb(0x44, 0xdd, 0xaa);
const CYCLONIC_COLOR: Color32 = Color32::from_rgb(0xff, 0x33, 0x33);
const ANTICYCLONIC_COLOR: Color32 = Color32::from_rgb(0x44, 0x88, 0xff);
const BORDER_COLOR: Color32 = Color32::from_rgb(0x33, 0x99, 0xcc);
const GRID_COLOR: Color32 = Color32::from_rgb(0x33, 0x50, 0x6a);
const LEGEND_FRAME_COLOR: Color32 = Color32::from_rgb(0x55, 0x5b, 0x62);

/// Streamwiseness samples on an evenly spaced height grid
/// (port of `StreamwisenessData`).
struct StreamwisenessData {
    height_km: Vec<f64>,
    percent: Vec<f64>,
    signed_percent: Vec<f64>,
}

/// Port of `streamwiseness_profile` (right mover, 0-6 km AGL, 100 m grid).
///
/// Horizontal vorticity is the horizontal part of `curl(V)` for a wind that
/// varies with height: `(-dv/dz, du/dz)`. Streamwiseness is the square of its
/// projection onto the storm-relative wind unit vector divided by the total
/// horizontal-vorticity magnitude. The sign of the projection is kept
/// separately for the cyclonic/anticyclonic shading.
///
/// In symbols, `percent = (omega_streamwise / |omega_h|)^2 * 100`. Squaring is
/// applied to the dimensionless ratio, not to a value that has already been
/// converted to percent.
fn streamwiseness_profile(prof: &Profile) -> Option<StreamwisenessData> {
    const MAX_HEIGHT_M: f64 = 6000.0;
    const STEP_M: f64 = 100.0;

    let inner = &prof.inner;
    let n = inner.hght.len().min(inner.u.len()).min(inner.v.len());
    let sfc = inner.sfc;
    if n < 2 || sfc >= n || !inner.hght[sfc].is_finite() {
        return None;
    }

    // Heights relative to the surface; keep only finite (h, u, v) triples.
    let mut height: Vec<f64> = Vec::new();
    let mut u: Vec<f64> = Vec::new();
    let mut v: Vec<f64> = Vec::new();
    for i in sfc..n {
        let h = inner.hght[i] - inner.hght[sfc];
        if h.is_finite() && inner.u[i].is_finite() && inner.v[i].is_finite() {
            height.push(h);
            u.push(inner.u[i]);
            v.push(inner.v[i]);
        }
    }
    if height.len() < 2 {
        return None;
    }

    // Stable sort by height, then drop duplicate heights (keep the first).
    let mut order: Vec<usize> = (0..height.len()).collect();
    order.sort_by(|&a, &b| height[a].partial_cmp(&height[b]).unwrap());
    let (hs, us, vs): (Vec<f64>, Vec<f64>, Vec<f64>) = (
        order.iter().map(|&i| height[i]).collect(),
        order.iter().map(|&i| u[i]).collect(),
        order.iter().map(|&i| v[i]).collect(),
    );
    let mut hu: Vec<(f64, f64, f64)> = Vec::with_capacity(hs.len());
    for i in 0..hs.len() {
        if hu.last().is_none_or(|&(h, _, _)| h != hs[i]) {
            hu.push((hs[i], us[i], vs[i]));
        }
    }
    if hu.len() < 2 {
        return None;
    }

    let top = MAX_HEIGHT_M.min(hu[hu.len() - 1].0);
    if top < STEP_M {
        return None;
    }
    let grid_top = (top / STEP_M).floor() * STEP_M;
    let npts = (grid_top / STEP_M).round() as usize + 1;
    if npts < 2 {
        return None;
    }
    let grid: Vec<f64> = (0..npts).map(|i| i as f64 * STEP_M).collect();

    // Bunkers right-mover storm motion (kts -> m/s).
    let (storm_u_kt, storm_v_kt) = (prof.srwind.0, prof.srwind.1);
    if !qc(storm_u_kt) || !qc(storm_v_kt) {
        return None;
    }
    let storm_u = storm_u_kt * KTS_TO_MS;
    let storm_v = storm_v_kt * KTS_TO_MS;

    // np.interp: clamp to the end values outside the data range.
    let interp = |x: f64, field: &dyn Fn(&(f64, f64, f64)) -> f64| -> f64 {
        if x <= hu[0].0 {
            return field(&hu[0]);
        }
        if x >= hu[hu.len() - 1].0 {
            return field(&hu[hu.len() - 1]);
        }
        let mut lo = 0;
        let mut hi = hu.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if hu[mid].0 <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (xa, xb) = (hu[lo].0, hu[hi].0);
        let (ya, yb) = (field(&hu[lo]), field(&hu[hi]));
        ya + (x - xa) / (xb - xa) * (yb - ya)
    };
    let u_ms: Vec<f64> = grid
        .iter()
        .map(|&z| interp(z, &|t| t.1) * KTS_TO_MS)
        .collect();
    let v_ms: Vec<f64> = grid
        .iter()
        .map(|&z| interp(z, &|t| t.2) * KTS_TO_MS)
        .collect();
    let u_sr: Vec<f64> = u_ms.iter().map(|&x| x - storm_u).collect();
    let v_sr: Vec<f64> = v_ms.iter().map(|&x| x - storm_v).collect();

    // np.gradient: central differences inside, one-sided at the ends.
    let gradient = |f: &[f64]| -> Vec<f64> {
        let m = f.len();
        let mut g = vec![0.0; m];
        g[0] = (f[1] - f[0]) / STEP_M;
        g[m - 1] = (f[m - 1] - f[m - 2]) / STEP_M;
        for i in 1..m - 1 {
            g[i] = (f[i + 1] - f[i - 1]) / (2.0 * STEP_M);
        }
        g
    };
    let dudz = gradient(&u_ms);
    let dvdz = gradient(&v_ms);

    let mut percent = vec![f64::NAN; npts];
    let mut signed = vec![f64::NAN; npts];
    let mut any_usable = false;
    for i in 0..npts {
        let omega_u = -dvdz[i];
        let omega_v = dudz[i];
        let omega_mag = omega_u.hypot(omega_v);
        let sr_speed = u_sr[i].hypot(v_sr[i]);
        if omega_mag > 1.0e-6 && sr_speed > 0.1 {
            any_usable = true;
            let omega_streamwise = omega_u * (u_sr[i] / sr_speed) + omega_v * (v_sr[i] / sr_speed);
            let (p, signed_p) = streamwiseness_percent(omega_streamwise, omega_mag);
            percent[i] = p;
            signed[i] = signed_p;
        }
    }
    if !any_usable {
        return None;
    }

    Some(StreamwisenessData {
        height_km: grid.iter().map(|&z| z / 1000.0).collect(),
        percent,
        signed_percent: signed,
    })
}

/// Return `(magnitude_percent, signed_percent)` for one usable sample.
fn streamwiseness_percent(omega_streamwise: f64, omega_mag: f64) -> (f64, f64) {
    let ratio = omega_streamwise / omega_mag;
    let percent = (ratio * ratio * 100.0).clamp(0.0, 100.0);
    (percent, omega_streamwise.signum() * percent)
}

/// Plot geometry (port of `plotStreamwiseness._geometry`), in rect-local px.
struct Geom {
    rect: Rect,
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
}

impl Geom {
    fn new(rect: Rect) -> Geom {
        let w = rect.width().max(1.0);
        let h = rect.height().max(1.0);
        let side_inset = 27.0f32.max((w * 0.14).floor());
        let left = side_inset;
        let mut right = w - RIGHT_INSET;
        let top = 22.0f32.max((h * 0.07).floor());
        let mut bottom = h - 30.0f32.max((h * 0.09).floor());
        if right <= left {
            right = left + 1.0;
        }
        if bottom <= top {
            bottom = top + 1.0;
        }
        Geom {
            rect,
            left,
            right,
            top,
            bottom,
        }
    }

    fn width(&self) -> f32 {
        self.right - self.left
    }

    fn height(&self) -> f32 {
        self.bottom - self.top
    }

    fn pt(&self, x: f32, y: f32) -> Pos2 {
        Pos2::new(self.rect.min.x + x, self.rect.min.y + y)
    }

    /// Port of `_x_to_pix`.
    fn x_to_pix(&self, value: f64) -> f32 {
        self.left + (value.clamp(0.0, 100.0) / 100.0) as f32 * self.width()
    }

    /// Port of `_y_to_pix`.
    fn y_to_pix(&self, height_km: f64) -> f32 {
        let fraction = height_km.clamp(0.0, MAX_HEIGHT_KM) / MAX_HEIGHT_KM;
        self.bottom - fraction as f32 * self.height()
    }
}

/// Draw the streamwiseness inset into `rect` (port of
/// `plotStreamwiseness._redraw`).
#[allow(unused_variables)]
pub fn draw(painter: &Painter, rect: Rect, prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) {
    let painter = painter.with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, style.bg_color);

    let g = Geom::new(rect);
    let h = rect.height().max(1.0);

    // The Qt original sets pixel sizes directly (QFont.setPixelSize), so no
    // pt -> px conversion here.
    let title_size = (h * 0.027).round().clamp(8.0, 11.0);
    let axis_size = (h * 0.022).round().clamp(7.0, 9.0);
    let tiny_size = (h * 0.019).round().clamp(6.0, 8.0);
    let title_font = style.bold_font(title_size);
    let axis_font = style.regular_font(axis_size);
    let axis_font_bold = style.bold_font(axis_size);
    let tiny_font = style.regular_font(tiny_size);
    let tiny_font_bold = style.bold_font(tiny_size);

    // Title, centered in the band above the plot.
    painter.text(
        g.pt(g.left + g.width() / 2.0, 2.0 + (g.top - 3.0) / 2.0),
        Align2::CENTER_CENTER,
        "Streamwiseness",
        title_font.clone(),
        style.fg_color,
    );

    draw_grid(&painter, &g, &axis_font, style);

    let data = streamwiseness_profile(prof);
    match &data {
        None => {
            painter.text(
                g.pt(g.left + g.width() / 2.0, g.top + g.height() / 2.0),
                Align2::CENTER_CENTER,
                "--",
                style.bold_font(12.0f32.max(title_size + 2.0)),
                style.fg_color,
            );
        }
        Some(data) => {
            draw_fills(&painter, &g, data);
            draw_profile(&painter, &g, data);
            draw_markers(&painter, &g, data, &tiny_font_bold, style);
            draw_legend(&painter, &g, style, &tiny_font);
        }
    }

    // Axis captions.
    let caption_h = 10.0f32.max(h - g.bottom - 13.0);
    painter.text(
        g.pt(g.left + g.width() / 2.0, g.bottom + 13.0 + caption_h / 2.0),
        Align2::CENTER_CENTER,
        "Streamwiseness (%)",
        axis_font_bold.clone(),
        style.fg_color,
    );
    // Rotated y-axis label, centered on (8, plot mid-height).
    let galley =
        painter.layout_no_wrap("Height AGL (km)".to_owned(), axis_font_bold, style.fg_color);
    let center = g.pt(8.0, g.top + g.height() / 2.0);
    let pos = center - Vec2::new(galley.size().x / 2.0, galley.size().y / 2.0);
    painter.add(Shape::Text(
        TextShape::new(pos, galley, style.fg_color)
            .with_angle_and_anchor(-std::f32::consts::FRAC_PI_2, Align2::CENTER_CENTER),
    ));

    // Left border line.
    painter.line_segment(
        [g.pt(0.5, 0.5), g.pt(0.5, (h - 0.5).max(0.5))],
        Stroke::new(1.0, BORDER_COLOR),
    );
}

/// Port of `_draw_grid`.
fn draw_grid(painter: &Painter, g: &Geom, axis_font: &FontId, style: &SkewTStyle) {
    let grid = Color32::from_rgba_unmultiplied(GRID_COLOR.r(), GRID_COLOR.g(), GRID_COLOR.b(), 130);
    let dashed = Stroke::new(1.0, grid);
    for tick in [0.0f64, 25.0, 50.0, 75.0, 100.0] {
        let x = g.x_to_pix(tick);
        if tick == 100.0 {
            painter.line_segment(
                [g.pt(x, g.top), g.pt(x, g.bottom)],
                Stroke::new(1.0, style.fg_color),
            );
        } else {
            painter.extend(Shape::dashed_line(
                &[g.pt(x, g.top), g.pt(x, g.bottom)],
                dashed,
                4.0,
                2.0,
            ));
        }
        if tick != 0.0 {
            painter.text(
                g.pt(x, g.bottom + 1.0 + 6.0),
                Align2::CENTER_CENTER,
                crate::utils::int2str(tick),
                axis_font.clone(),
                style.fg_color,
            );
        }
    }
    for tick in 0..=6 {
        let y = g.y_to_pix(tick as f64);
        painter.extend(Shape::dashed_line(
            &[g.pt(g.left, y), g.pt(g.right, y)],
            dashed,
            4.0,
            2.0,
        ));
        // Label rect: (11, y-6, max(14, left-13), 12), right-aligned.
        painter.text(
            g.pt(11.0 + 14.0f32.max(g.left - 13.0), y),
            Align2::RIGHT_CENTER,
            format!("{tick}"),
            axis_font.clone(),
            style.fg_color,
        );
    }
}

/// Port of `_draw_fills`: per-segment cyclonic/anticyclonic shading.
fn draw_fills(painter: &Painter, g: &Geom, data: &StreamwisenessData) {
    for i in 0..data.height_km.len().saturating_sub(1) {
        let (p0, p1) = (data.percent[i], data.percent[i + 1]);
        let (s0, s1) = (data.signed_percent[i], data.signed_percent[i + 1]);
        if !(p0.is_finite() && p1.is_finite() && s0.is_finite() && s1.is_finite()) {
            continue;
        }
        let y0 = g.y_to_pix(data.height_km[i]);
        let y1 = g.y_to_pix(data.height_km[i + 1]);
        let x0 = g.x_to_pix(p0);
        let x1 = g.x_to_pix(p1);
        let base = if (s0 + s1) >= 0.0 {
            CYCLONIC_COLOR
        } else {
            ANTICYCLONIC_COLOR
        };
        let fill = Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 52);
        painter.add(Shape::convex_polygon(
            vec![
                g.pt(g.left, y0),
                g.pt(x0, y0),
                g.pt(x1, y1),
                g.pt(g.left, y1),
            ],
            fill,
            Stroke::NONE,
        ));
    }
}

/// Port of `_draw_profile`: the green streamwiseness-% curve.
fn draw_profile(painter: &Painter, g: &Geom, data: &StreamwisenessData) {
    let stroke = Stroke::new(2.0, PROFILE_COLOR);
    let mut run: Vec<Pos2> = Vec::new();
    for (&value, &height_km) in data.percent.iter().zip(&data.height_km) {
        if !value.is_finite() || height_km > MAX_HEIGHT_KM {
            if run.len() >= 2 {
                painter.add(Shape::line(std::mem::take(&mut run), stroke));
            } else {
                run.clear();
            }
            continue;
        }
        run.push(g.pt(g.x_to_pix(value), g.y_to_pix(height_km)));
    }
    if run.len() >= 2 {
        painter.add(Shape::line(run, stroke));
    }
}

/// Port of `_draw_markers`: dashed threshold lines + dots + % labels at
/// 0.5 / 1 / 3 km AGL.
fn draw_markers(
    painter: &Painter,
    g: &Geom,
    data: &StreamwisenessData,
    font: &FontId,
    style: &SkewTStyle,
) {
    for (depth, color) in [
        (0.5, Color32::from_rgb(0xb8, 0xbc, 0xc2)),
        (1.0, Color32::from_rgb(0xff, 0x88, 0x00)),
        (3.0, Color32::from_rgb(0xff, 0xcc, 0x00)),
    ] {
        // Nearest grid level (among finite samples) to the target depth.
        let mut best: Option<(f64, usize)> = None;
        for i in 0..data.percent.len() {
            if !data.percent[i].is_finite() {
                continue;
            }
            let d = (data.height_km[i] - depth).abs();
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, i));
            }
        }
        let Some((_, index)) = best else { continue };
        let value = data.percent[index];
        if !value.is_finite() {
            continue;
        }
        let color_dim = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 115);
        let y = g.y_to_pix(depth);
        let x = g.x_to_pix(value);
        painter.extend(Shape::dashed_line(
            &[g.pt(g.left, y), g.pt(g.right, y)],
            Stroke::new(1.0, color_dim),
            4.0,
            2.0,
        ));
        painter.circle(g.pt(x, y), 2.8, color, Stroke::new(1.0, style.fg_color));
        let label = format!("{value:.0}%");
        if x > g.right - 35.0 {
            painter.text(
                g.pt(x - 3.0, y - 7.0),
                Align2::RIGHT_CENTER,
                label,
                font.clone(),
                style.fg_color,
            );
        } else {
            painter.text(
                g.pt(x + 4.0, y - 7.0),
                Align2::LEFT_CENTER,
                label,
                font.clone(),
                style.fg_color,
            );
        }
    }
}

/// Port of `_draw_legend`: Cyclonic / Anticyclonic swatch box, top-right.
fn draw_legend(painter: &Painter, g: &Geom, style: &SkewTStyle, font: &FontId) {
    let labels = [
        ("Cyclonic", CYCLONIC_COLOR),
        ("Anticyclonic", ANTICYCLONIC_COLOR),
    ];
    let mut max_adv = 0.0f32;
    let mut font_h = 0.0f32;
    for (label, _) in labels {
        let galley = painter.layout_no_wrap(label.to_owned(), font.clone(), style.fg_color);
        max_adv = max_adv.max(galley.size().x);
        font_h = font_h.max(galley.size().y);
    }
    let row_h = 9.0f32.max(font_h);
    let width = (g.width() - 4.0).min(max_adv + 18.0 + 4.0);
    let height = row_h * labels.len() as f32 + 4.0;
    let left = g.right - width - 2.0;
    let top = g.top + 2.0;
    let legend_rect = Rect::from_min_size(g.pt(left, top), Vec2::new(width, height));
    let background = Color32::from_rgba_unmultiplied(
        style.bg_color.r(),
        style.bg_color.g(),
        style.bg_color.b(),
        220,
    );
    painter.rect(
        legend_rect,
        0.0,
        background,
        Stroke::new(1.0, LEGEND_FRAME_COLOR),
        StrokeKind::Middle,
    );
    for (row, (label, color)) in labels.iter().enumerate() {
        let y = top + 2.0 + row as f32 * row_h;
        let fill = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 90);
        painter.rect(
            Rect::from_min_size(g.pt(left + 3.0, y + 2.0), Vec2::new(10.0, row_h - 4.0)),
            0.0,
            fill,
            Stroke::new(1.0, *color),
            StrokeKind::Middle,
        );
        painter.text(
            g.pt(left + 16.0, y + row_h / 2.0),
            Align2::LEFT_CENTER,
            *label,
            font.clone(),
            style.fg_color,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::streamwiseness_percent;

    #[test]
    fn half_streamwise_vorticity_is_twenty_five_percent() {
        let (percent, signed) = streamwiseness_percent(1.0, 2.0);

        assert_eq!(percent, 25.0);
        assert_eq!(signed, 25.0);
    }

    #[test]
    fn negative_half_streamwise_vorticity_keeps_negative_shading() {
        let (percent, signed) = streamwiseness_percent(-1.0, 2.0);

        assert_eq!(percent, 25.0);
        assert_eq!(signed, -25.0);
    }
}
