//! "Sig Hail Param (SHIP)" box-and-whisker inset — port of the `_ship_chart`
//! block of `sharpmod/viz/index_board.py` (SHARPpy-Reimagined). The two hail
//! climatology distributions come from `sharppy.databases.inset_data.shipData`
//! (developed internally by Ryan Jewell, SPC): a "< 2 in" (yellow) and a
//! ">= 2 in" (red) box-whisker on a 0-5 SHIP scale with dashed horizontal
//! gridlines, plus the current SHIP value drawn as a full-width marker line
//! colored by the hail-size class it falls in (red once SHIP >= 1, yellow
//! below — the same yellow/red scheme as the categories).

use egui::{Align2, Color32, Painter, Rect, Shape, Stroke, StrokeKind, pos2};

use crate::Profile;
use crate::derived::DerivedParams;
use crate::skewt::SkewTStyle;
use crate::utils::qc;

/// `[whisker_lo, box_bottom, median, box_top, whisker_hi]` per hail category
/// (`inset_data.shipData()["ship_dist"]`).
const SHIP_DIST: [[f64; 5]; 2] = [
    [0.2, 0.3, 0.2, 0.9, 1.2], // "< 2 in"
    [1.1, 1.4, 0.8, 2.8, 4.0], // ">= 2 in"
];
/// Category labels; the trailing " in" is drawn in the smaller unit font.
const CAT_VALUES: [&str; 2] = ["< 2", ">= 2"];
/// "< 2 in" -> yellow, ">= 2 in" -> red (matching the STP EF-scale look).
const CAT_COLORS: [Color32; 2] = [
    Color32::from_rgb(0xFF, 0xFF, 0x00),
    Color32::from_rgb(0xFF, 0x00, 0x00),
];
const GRID_COLOR: Color32 = Color32::from_rgb(0x2F, 0x6D, 0x88);

/// Draw the SHIP box-whisker chart into `rect`.
pub fn draw(painter: &Painter, rect: Rect, prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) {
    let _ = prof; // the SHIP value is read from the derived params
    if rect.width() < 40.0 || rect.height() < 40.0 {
        return;
    }
    let p = painter.with_clip_rect(rect);
    p.rect_filled(rect, 0.0, style.bg_color);

    let (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height());
    // Small bold header font, sized from the inset height (the Qt original
    // used the board's 10-px bold header font over a ~150-px-tall chart).
    let fs = (h * 0.062).clamp(8.0, 40.0);
    let font = style.bold_font(fs);
    let unit_font = style.bold_font((fs * 0.78).round().max(8.0));

    p.text(
        pos2(x + w / 2.0, y + 1.0),
        Align2::CENTER_TOP,
        "Sig Hail Param (SHIP)",
        font.clone(),
        style.fg_color,
    );

    let top = y + fs * 1.4;
    let bottom = y + h - fs * 1.2; // leave room for the x-axis labels
    if bottom - top < 20.0 {
        return;
    }
    // SHIP value (clamped to [0, 5]) -> y pixel.
    let toy = |v: f64| -> f32 {
        let frac = (v.clamp(0.0, 5.0) / 5.0) as f32;
        bottom - frac * (bottom - top)
    };

    let lbl_w = p
        .layout_no_wrap("0".to_owned(), font.clone(), style.fg_color)
        .size()
        .x
        + 2.0;
    let ax0 = x + lbl_w + 4.0;
    let ax1 = x + w - 3.0;

    // Dashed gridlines + y-axis labels 0..5.
    let gsw = (h * 0.005).clamp(1.0, 2.5);
    for gv in 0..=5 {
        let gy = toy(gv as f64);
        p.extend(Shape::dashed_line(
            &[pos2(ax0, gy), pos2(ax1, gy)],
            Stroke::new(gsw, GRID_COLOR),
            4.0 * gsw,
            2.5 * gsw,
        ));
        p.text(
            pos2(x + lbl_w, gy),
            Align2::RIGHT_CENTER,
            gv.to_string(),
            font.clone(),
            style.fg_color,
        );
    }

    // Box-and-whisker per hail category.
    let n = SHIP_DIST.len();
    let plotw = ax1 - ax0;
    let sw = (h * 0.010).clamp(1.5, 4.0);
    for i in 0..n {
        let [wl, bb, med, bt, wh] = SHIP_DIST[i];
        let cx = ax0 + (i as f32 + 0.5) * plotw / n as f32;
        let bw = (plotw / n as f32 * 0.28).max(6.0);
        let cc = CAT_COLORS[i];
        let stroke = Stroke::new(sw, cc);
        // Whiskers.
        p.line_segment([pos2(cx, toy(wl.min(bb))), pos2(cx, toy(bb))], stroke);
        p.line_segment([pos2(cx, toy(bt)), pos2(cx, toy(wh.max(bt)))], stroke);
        // Box outline + median line.
        p.rect_stroke(
            Rect::from_two_pos(pos2(cx - bw, toy(bt)), pos2(cx + bw, toy(bb))),
            0.0,
            stroke,
            StrokeKind::Middle,
        );
        p.line_segment([pos2(cx - bw, toy(med)), pos2(cx + bw, toy(med))], stroke);
        // Category label beneath, with " in" in the smaller unit font.
        let vw = p
            .layout_no_wrap(CAT_VALUES[i].to_owned(), font.clone(), cc)
            .size()
            .x;
        let uw = p
            .layout_no_wrap(" in".to_owned(), unit_font.clone(), cc)
            .size()
            .x;
        let left = cx - (vw + uw) / 2.0;
        let ly = bottom + 2.0 + fs * 0.62;
        p.text(
            pos2(left, ly),
            Align2::LEFT_CENTER,
            CAT_VALUES[i],
            font.clone(),
            cc,
        );
        p.text(
            pos2(left + vw, ly),
            Align2::LEFT_CENTER,
            " in",
            unit_font.clone(),
            cc,
        );
    }

    // Current SHIP value as a reference line, red once it reaches the
    // sig-hail (>= 2 in) regime (SHIP >= 1), yellow below.
    if qc(dv.ship) {
        let sy = toy(dv.ship);
        let col = if dv.ship >= 1.0 {
            CAT_COLORS[1]
        } else {
            CAT_COLORS[0]
        };
        p.line_segment([pos2(ax0, sy), pos2(ax1, sy)], Stroke::new(sw, col));
    }
}
