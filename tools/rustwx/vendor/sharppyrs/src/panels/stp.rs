//! "Effective Layer STP (with CIN)" inset — port of the vendored
//! `sharppy.viz.stp` (`backgroundSTP` + `plotSTP`) with the
//! SHARPpy-Reimagined patches from `sharpmod.render` applied:
//! per-EF-category box/label colors, the `NONTOR -> NON` rename, the
//! content-fitted conditional-probability box, `STP_LABEL_SCALE` (0.72) on the
//! tick/EF label fonts, and `STP_BOTTOM_MARGIN` (16 px) so the EF labels do
//! not clip at the bottom edge.

use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Shape, Stroke, StrokeKind};

use crate::Profile;
use crate::derived::DerivedParams;
use crate::skewt::SkewTStyle;
use crate::utils::qc;

const PT: f32 = 4.0 / 3.0;
const FONT_RATIO: f32 = 0.0512;
const TEXTPAD: f32 = 5.0;
const STP_MAX: f64 = 11.0;

/// `sharpmod.render.STP_LABEL_SCALE` default.
const STP_LABEL_SCALE: f32 = 0.72;
/// `sharpmod.render.STP_BOTTOM_MARGIN` default.
const STP_BOTTOM_MARGIN: f32 = 16.0;

const BORDER_COLOR: Color32 = Color32::from_rgb(0x33, 0x99, 0xCC);
const BOX_COLOR: Color32 = Color32::from_rgb(0x00, 0xFF, 0x00);
const LINE_COLOR: Color32 = Color32::from_rgb(0x00, 0x80, 0xFF);

/// STP box/whisker climatology from Thompson et al. 2012 WAF
/// (`sharppy.databases.inset_data.stpData`), rows EF4+ .. NONTOR, columns
/// `[whisker lo, box bottom, median, box top, whisker hi]`.
const EF_CLIMO: [[f64; 5]; 6] = [
    [1.2, 2.6, 5.3, 8.3, 11.0], // EF4+
    [0.2, 1.0, 2.4, 4.5, 8.4],  // EF3
    [0.0, 0.6, 1.7, 3.7, 5.6],  // EF2
    [0.0, 0.3, 1.2, 2.6, 4.5],  // EF1
    [0.0, 0.1, 0.8, 2.0, 3.7],  // EF0
    [0.0, 0.0, 0.2, 0.7, 1.7],  // NONTOR ("NON")
];

/// X-axis labels ("NONTOR" renamed "NON" by `_install_stp_label_rename`) and
/// the per-EF colors of `sharpmod.render.STP_XLABEL_COLORS` (`None` = keep the
/// vendored green box / foreground label).
const EF_LABELS: [(&str, Option<Color32>); 6] = [
    ("EF4+", Some(Color32::from_rgb(0xFF, 0x00, 0xFF))),
    ("EF3", Some(Color32::from_rgb(0xFF, 0x00, 0x00))),
    ("EF2", Some(Color32::from_rgb(0xFF, 0xA5, 0x00))),
    ("EF1", Some(Color32::from_rgb(0xFF, 0xFF, 0x00))),
    ("EF0", Some(Color32::from_rgb(0x33, 0x99, 0xFF))),
    ("NON", None),
];

const PROB_HEADERS: [&str; 2] = ["Prob EF2+ torn with supercell", "Sample CLIMO = .15 sigtor"];
const PROB_LABELS: [&str; 6] = [
    "based on CAPE:",
    "based on LCL:",
    "based on ESRH:",
    "based on EBWD:",
    "based on STPC:",
    "based on STP_fixed:",
];

/// Geometry (port of `backgroundSTP.initUI` + the `STP_BOTTOM_MARGIN` patch),
/// in rect-local px.
struct Geom {
    rect: Rect,
    tpad: f32,
    bpad: f32,
    wid: f32,
    brx: f32,
    bry: f32,
    hgt: f32,
}

impl Geom {
    fn new(rect: Rect, plot_height: f32) -> Geom {
        let w = rect.width().max(1.0);
        let h = rect.height().max(1.0);
        let tpad = plot_height + 15.0;
        let bpad = plot_height + 2.0 + STP_BOTTOM_MARGIN;
        let hgt = h - bpad;
        Geom {
            rect,
            tpad,
            bpad,
            wid: w,
            brx: w,
            bry: hgt - bpad,
            hgt,
        }
    }

    fn pt(&self, x: f32, y: f32) -> Pos2 {
        Pos2::new(self.rect.min.x + x, self.rect.min.y + y)
    }

    /// Port of `stp_to_pix`.
    fn stp_to_pix(&self, stp: f64) -> f32 {
        self.bry - (stp / STP_MAX) as f32 * (self.bry - self.tpad)
    }
}

/// The vendored widget builds Qt point-size fonts from the widget height and,
/// on Windows, immediately trims the font-metrics descent off the point size.
/// Model that trim (descent of Helvetica ~= 0.21 em of the pixel size).
fn windows_descent_trim(pt: f32) -> f32 {
    (pt - (pt * PT * 0.21).round()).max(1.0)
}

/// `str(round(val, 2))`-style formatting (port of `tab.utils.FLOAT2STR`,
/// which drops trailing zeros: 0.20 -> "0.2").
fn float2str_py(val: f64, precision: usize) -> String {
    if !qc(val) {
        return "--".to_string();
    }
    let s = format!("{val:.precision$}");
    if !s.contains('.') {
        return s;
    }
    let mut t = s.trim_end_matches('0').to_string();
    if t.ends_with('.') {
        t.push('0');
    }
    t
}

// ----------------------------------------------------------------------
// Conditional EF2+ tornado probabilities (ports of plotSTP.*_prob).
// Return (probability, alert-color index into style.alert_colors).
// Some adjacent climatology bins share a value; they are kept separate so
// the tables stay 1:1 with the Python source.
// ----------------------------------------------------------------------

#[allow(clippy::if_same_then_else)]
fn cape_prob(cape: f64) -> (f64, usize) {
    if cape == 0.0 {
        (0.00, 0)
    } else if cape > 0.0 && cape < 250.0 {
        (0.12, 1)
    } else if (250.0..500.0).contains(&cape) {
        (0.14, 2)
    } else if (500.0..1000.0).contains(&cape) {
        (0.16, 2)
    } else if (1000.0..1500.0).contains(&cape) {
        (0.15, 2)
    } else if (1500.0..2000.0).contains(&cape) {
        (0.13, 2)
    } else if (2000.0..2500.0).contains(&cape) {
        (0.14, 2)
    } else if (2500.0..3000.0).contains(&cape) {
        (0.18, 3)
    } else if (3000.0..4000.0).contains(&cape) {
        (0.20, 3)
    } else if cape >= 4000.0 {
        (0.16, 3)
    } else {
        (f64::NAN, 0)
    }
}

#[allow(clippy::if_same_then_else)]
fn lcl_prob(lcl: f64) -> (f64, usize) {
    if lcl < 750.0 {
        (0.19, 3)
    } else if (750.0..1000.0).contains(&lcl) {
        (0.19, 3)
    } else if (1000.0..1250.0).contains(&lcl) {
        (0.15, 2)
    } else if (1250.0..1500.0).contains(&lcl) {
        (0.10, 1)
    } else if (1500.0..1750.0).contains(&lcl) {
        (0.06, 0)
    } else if (1750.0..2000.0).contains(&lcl) {
        (0.06, 0)
    } else if (2000.0..2500.0).contains(&lcl) {
        (0.02, 0)
    } else if lcl >= 2500.0 {
        (0.0, 0)
    } else {
        (f64::NAN, 0)
    }
}

#[allow(clippy::if_same_then_else)]
fn esrh_prob(esrh: f64) -> (f64, usize) {
    if esrh < 50.0 {
        (0.06, 0)
    } else if (50.0..100.0).contains(&esrh) {
        (0.06, 0)
    } else if (100.0..200.0).contains(&esrh) {
        (0.08, 1)
    } else if (200.0..300.0).contains(&esrh) {
        (0.14, 2)
    } else if (300.0..400.0).contains(&esrh) {
        (0.20, 3)
    } else if (400.0..500.0).contains(&esrh) {
        (0.27, 3)
    } else if (500.0..600.0).contains(&esrh) {
        (0.38, 4)
    } else if (600.0..700.0).contains(&esrh) {
        (0.37, 4)
    } else if esrh >= 700.0 {
        (0.42, 4)
    } else {
        (f64::NAN, 0)
    }
}

fn ebwd_prob(ebwd: f64) -> (f64, usize) {
    if ebwd == 0.0 {
        (0.0, 0)
    } else if (0.01..20.0).contains(&ebwd) {
        (0.03, 0)
    } else if (20.0..30.0).contains(&ebwd) {
        (0.05, 0)
    } else if (30.0..40.0).contains(&ebwd) {
        (0.06, 0)
    } else if (40.0..50.0).contains(&ebwd) {
        (0.12, 1)
    } else if (50.0..60.0).contains(&ebwd) {
        (0.19, 3)
    } else if (60.0..70.0).contains(&ebwd) {
        (0.27, 3)
    } else if (70.0..80.0).contains(&ebwd) {
        (0.36, 4)
    } else if ebwd >= 80.0 {
        (0.26, 3)
    } else {
        (f64::NAN, 0)
    }
}

fn stpc_prob(stpc: f64) -> (f64, usize) {
    if stpc < 0.1 {
        (0.06, 0)
    } else if (0.1..0.5).contains(&stpc) {
        (0.08, 1)
    } else if (0.5..1.0).contains(&stpc) {
        (0.12, 1)
    } else if (1.0..2.0).contains(&stpc) {
        (0.17, 2)
    } else if (2.0..4.0).contains(&stpc) {
        (0.25, 3)
    } else if (4.0..6.0).contains(&stpc) {
        (0.32, 4)
    } else if (6.0..8.0).contains(&stpc) {
        (0.34, 4)
    } else if (8.0..10.0).contains(&stpc) {
        (0.55, 5)
    } else if stpc >= 10.0 {
        (0.58, 5)
    } else {
        (f64::NAN, 0)
    }
}

#[allow(clippy::if_same_then_else)]
fn stpf_prob(stpf: f64) -> (f64, usize) {
    if stpf < 0.1 {
        (0.05, 0)
    } else if (0.1..0.5).contains(&stpf) {
        (0.06, 0)
    } else if (0.5..1.0).contains(&stpf) {
        (0.11, 1)
    } else if (1.0..2.0).contains(&stpf) {
        (0.17, 2)
    } else if (2.0..3.0).contains(&stpf) {
        (0.25, 3)
    } else if (3.0..5.0).contains(&stpf) {
        (0.25, 3)
    } else if (5.0..7.0).contains(&stpf) {
        (0.39, 4)
    } else if (7.0..9.0).contains(&stpf) {
        (0.55, 5)
    } else if stpf >= 9.0 {
        (0.59, 5)
    } else {
        (f64::NAN, 0)
    }
}

/// Draw the Effective Layer STP inset into `rect`.
pub fn draw(painter: &Painter, rect: Rect, prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) {
    let painter = painter.with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, style.bg_color);

    let h = rect.height().max(1.0);
    // fsize1/fsize2 from initUI (point sizes; Windows trims the descent off).
    let fsize1 = windows_descent_trim((h * FONT_RATIO).round() + 2.0);
    let fsize2 = windows_descent_trim((h * FONT_RATIO).round());
    let plot_font = style.regular_font(fsize1 * PT);
    // plot_height = plot_metrics.xHeight() (~0.52 em of the pixel size).
    let plot_height = fsize1 * PT * 0.52;
    let g = Geom::new(rect, plot_height);

    draw_frame(&painter, &g, style, &plot_font, plot_height);

    // --- current values (port of plotSTP.setProf, right mover) ---
    let mlcape = prof.mlpcl.bplus;
    let mllcl = prof.mlpcl.lclhght;
    let ebwspd = if dv.ebwd.0.is_finite() && dv.ebwd.1.is_finite() {
        dv.ebwd.0.hypot(dv.ebwd.1)
    } else {
        f64::NAN
    };
    let mut esrh = prof.right_esrh;
    let mut stpc = dv.stp_cin;
    let mut stpf = dv.stp_fixed;
    if prof.latitude() < 0.0 {
        esrh = -esrh;
        stpc = -stpc;
        stpf = -stpf;
    }

    draw_stp(&painter, &g, style, stpc);
    draw_prob_box(
        &painter,
        &g,
        style,
        fsize2,
        [
            cape_prob(mlcape),
            lcl_prob(mllcl),
            esrh_prob(esrh),
            ebwd_prob(ebwspd),
            stpc_prob(stpc),
            stpf_prob(stpf),
        ],
    );

    // Widget frame (the Qt stylesheet's 1px solid #3399CC border).
    painter.rect_stroke(
        rect.shrink(0.5),
        0.0,
        Stroke::new(1.0, BORDER_COLOR),
        StrokeKind::Inside,
    );
}

/// Background frame: title, dashed y gridlines + tick labels, and the
/// per-EF-colored box-and-whisker plots with their x labels (port of the
/// `_install_stp_xlabel_colors` replacement `draw_frame`).
fn draw_frame(
    painter: &Painter,
    g: &Geom,
    style: &SkewTStyle,
    plot_font: &FontId,
    plot_height: f32,
) {
    // Title, centered in QRectF(0, 5, brx, plot_height).
    painter.text(
        g.pt(g.brx / 2.0, 5.0 + plot_height / 2.0),
        Align2::CENTER_CENTER,
        "Effective Layer STP (with CIN)",
        plot_font.clone(),
        style.fg_color,
    );

    // Y gridlines (dashed, line_color) + white tick labels 11..0. The label
    // scale patch shrinks font_ratio for these draw-time fonts.
    let ytick_size = ((STP_LABEL_SCALE * FONT_RATIO * g.hgt).round() + 1.0) * PT;
    let ytick_font = style.regular_font(ytick_size);
    for yt in (0..=11).rev() {
        let tick_pxl = g.stp_to_pix(yt as f64);
        painter.extend(Shape::dashed_line(
            &[g.pt(0.0, tick_pxl), g.pt(g.brx, tick_pxl)],
            Stroke::new(1.0, LINE_COLOR),
            4.0,
            2.0,
        ));
        // Label rect QRect(tlx, tick - fs/2, 20, fs), centered.
        painter.text(
            g.pt(10.0, tick_pxl),
            Align2::CENTER_CENTER,
            format!("{yt}"),
            ytick_font.clone(),
            style.fg_color,
        );
    }

    // Box-and-whisker per EF category.
    let width = g.brx / 14.0;
    let spacing = g.brx / 7.0;
    let ef_size = (STP_LABEL_SCALE * FONT_RATIO * g.hgt).round() * PT;
    let ef_font = style.regular_font(ef_size);
    for (i, (row, (text, ef_color))) in EF_CLIMO.iter().zip(EF_LABELS).enumerate() {
        let cx = spacing * (i as f32 + 1.0);
        if cx >= g.brx {
            break;
        }
        let box_color = ef_color.unwrap_or(BOX_COLOR);
        let label_color = ef_color.unwrap_or(style.fg_color);
        draw_whisker_box(painter, g, cx, width, row, Stroke::new(2.0, box_color));
        // Label rect QRectF(cx - w/2, bry + round(bpad/2), w, bpad), centered.
        painter.text(
            g.pt(cx, g.bry + (g.bpad / 2.0).round() + g.bpad / 2.0),
            Align2::CENTER_CENTER,
            text,
            ef_font.clone(),
            label_color,
        );
    }
}

/// One box-and-whisker (port of `_draw_box` in the render patch): lower
/// whisker, box top/bottom/sides, median, upper whisker.
fn draw_whisker_box(
    painter: &Painter,
    g: &Geom,
    cx: f32,
    width: f32,
    row: &[f64; 5],
    stroke: Stroke,
) {
    let wl = g.stp_to_pix(row[0]);
    let bb = g.stp_to_pix(row[1]);
    let med = g.stp_to_pix(row[2]);
    let bt = g.stp_to_pix(row[3]);
    let wh = g.stp_to_pix(row[4]);
    let hw = width / 2.0;
    painter.line_segment([g.pt(cx, wl), g.pt(cx, bb)], stroke);
    painter.line_segment([g.pt(cx - hw, bt), g.pt(cx + hw, bt)], stroke);
    painter.line_segment([g.pt(cx - hw, bb), g.pt(cx + hw, bb)], stroke);
    painter.line_segment([g.pt(cx - hw, bb), g.pt(cx - hw, bt)], stroke);
    painter.line_segment([g.pt(cx + hw, bb), g.pt(cx + hw, bt)], stroke);
    painter.line_segment([g.pt(cx - hw, med), g.pt(cx + hw, med)], stroke);
    painter.line_segment([g.pt(cx, bt), g.pt(cx, wh)], stroke);
}

/// The current-STP marker line across the chart, colored by its conditional
/// probability alert level (port of `plotSTP.draw_stp`).
fn draw_stp(painter: &Painter, g: &Geom, style: &SkewTStyle, stpc: f64) {
    if !qc(stpc) {
        return;
    }
    let clamped = stpc.clamp(0.0, STP_MAX);
    let (_, color_idx) = stpc_prob(clamped);
    let y = g.stp_to_pix(clamped);
    painter.line_segment(
        [g.pt(0.0, y), g.pt(g.wid, y)],
        Stroke::new(1.5, style.alert_colors[color_idx]),
    );
}

/// The "Prob EF2+ torn with supercell" text block, top right (port of the
/// `_install_stp_prob_box_spacing` replacement `draw_box`: content-fitted
/// width, symmetric divider gap, consistent row height).
fn draw_prob_box(
    painter: &Painter,
    g: &Geom,
    style: &SkewTStyle,
    fsize2: f32,
    probs: [(f64, usize); 6],
) {
    let width = g.brx / 14.0;
    let top_y = g.stp_to_pix(STP_MAX);

    // Measure the content with the box font; shrink the font to fit the
    // available right half if needed.
    let measure = |font: &FontId, text: &str| -> f32 {
        painter
            .layout_no_wrap(text.to_owned(), font.clone(), style.fg_color)
            .size()
            .x
    };
    let mut box_pt = fsize2;
    let mut box_font = style.regular_font(box_pt * PT);
    let content_width = |font: &FontId| -> (f32, f32, f32) {
        let label_w = PROB_LABELS
            .iter()
            .map(|t| measure(font, t))
            .fold(0.0f32, f32::max);
        let col_gap = 12.0f32.max(measure(font, "  "));
        let val_w = measure(font, "0.00");
        let header_w = PROB_HEADERS
            .iter()
            .map(|t| measure(font, t))
            .fold(0.0f32, f32::max);
        ((label_w + col_gap + val_w).max(header_w), label_w, col_gap)
    };
    let (mut content_w, mut label_w, mut col_gap) = content_width(&box_font);
    let available_w = 40.0f32.max(g.brx - width * 7.0 - 10.0);
    if content_w + 8.0 > available_w {
        let scale = available_w / (content_w + 8.0);
        box_pt = (box_pt * scale * 0.96).max(5.0);
        box_font = style.regular_font(box_pt * PT);
        (content_w, label_w, col_gap) = content_width(&box_font);
    }

    // Anchor against the right edge, but never left of the mid-line.
    let right_x = g.brx - 5.0;
    let left_x = (width * 7.0).max(right_x - (content_w + 8.0));

    // Row metrics: box_height = xHeight + textpad; row_h adds 1 px plus the
    // font descent (the Windows branch of the original).
    let px = box_pt * PT;
    let box_height = px * 0.52 + TEXTPAD;
    let row_h = box_height + 1.0 + px * 0.21;
    let div_gap = 3.0f32.max((row_h * 0.4).round());
    let bot_y = top_y + 2.0 + 8.0 * row_h + div_gap + 2.0;

    // Black-filled box with a 2px foreground border.
    let box_rect = Rect::from_min_max(g.pt(left_x, top_y), g.pt(right_x, bot_y));
    painter.rect_filled(box_rect, 0.0, style.bg_color);
    painter.rect_stroke(
        box_rect,
        0.0,
        Stroke::new(2.0, style.fg_color),
        StrokeKind::Middle,
    );

    let x1 = left_x + 3.0;
    let x2 = x1 + label_w + col_gap;
    let mut y1 = top_y + 2.0;

    // Header rows.
    for text in PROB_HEADERS {
        painter.text(
            g.pt(x1, y1),
            Align2::LEFT_TOP,
            text,
            box_font.clone(),
            style.fg_color,
        );
        y1 += row_h;
    }

    // Divider rule, centred in its gap.
    let div_y = y1 + div_gap / 2.0;
    painter.line_segment(
        [g.pt(left_x, div_y), g.pt(right_x, div_y)],
        Stroke::new(1.0, style.fg_color),
    );
    y1 += div_gap;

    // Variable rows, label + probability in the row's alert color.
    for (text, (p, color_idx)) in PROB_LABELS.iter().zip(probs) {
        let color = style.alert_colors[color_idx];
        painter.text(
            g.pt(x1, y1),
            Align2::LEFT_TOP,
            *text,
            box_font.clone(),
            color,
        );
        painter.text(
            g.pt(x2, y1),
            Align2::LEFT_TOP,
            float2str_py(p, 2),
            box_font.clone(),
            color,
        );
        y1 += row_h;
    }
}
