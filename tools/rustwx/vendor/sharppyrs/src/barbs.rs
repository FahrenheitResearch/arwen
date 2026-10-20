//! Wind barb drawing with the SHARPpy-Reimagined speed color table
//! (port of `sharpmod.viz.custom_barbs`).

use egui::{Color32, Painter, Pos2, Stroke, Vec2};

/// Speed (kt) -> color table, evaluated high-to-low so the last matching
/// branch for a given speed wins. >= 80 kt is a pink->white gradient.
const BARB_TABLE: [(f64, Color32); 18] = [
    (100.0, Color32::from_rgb(0xFF, 0xE6, 0xF2)),
    (95.0, Color32::from_rgb(0xFF, 0xCC, 0xE5)),
    (90.0, Color32::from_rgb(0xFF, 0xB3, 0xD9)),
    (85.0, Color32::from_rgb(0xFF, 0x99, 0xCC)),
    (80.0, Color32::from_rgb(0xFF, 0x80, 0xBF)),
    (75.0, Color32::from_rgb(0xFF, 0x00, 0x00)),
    (60.0, Color32::from_rgb(0xFF, 0x40, 0x00)),
    (55.0, Color32::from_rgb(0xFF, 0x80, 0x00)),
    (50.0, Color32::from_rgb(0xFF, 0xBF, 0x00)),
    (45.0, Color32::from_rgb(0xFF, 0xFF, 0x00)),
    (40.0, Color32::from_rgb(0xBF, 0xFF, 0x00)),
    (35.0, Color32::from_rgb(0x80, 0xFF, 0x00)),
    (30.0, Color32::from_rgb(0x40, 0xFF, 0x00)),
    (25.0, Color32::from_rgb(0x00, 0xFF, 0x00)),
    (20.0, Color32::from_rgb(0x00, 0xFF, 0x40)),
    (15.0, Color32::from_rgb(0x00, 0xFF, 0x80)),
    (10.0, Color32::from_rgb(0x00, 0xFF, 0xBF)),
    (5.0, Color32::from_rgb(0x00, 0xFF, 0xFF)),
];

/// The barb color for a wind speed (kt).
pub fn barb_color(wspd: f64) -> Color32 {
    let mut color = Color32::WHITE; // > 100 kt reads white
    for (threshold, c) in BARB_TABLE {
        if wspd <= threshold {
            color = c;
        }
    }
    if wspd < 3.0 {
        color = Color32::WHITE;
    }
    color
}

/// Draw a wind barb at `origin` for direction `wdir` (deg) / speed `wspd`
/// (kt), colored by the speed table. `shemis` mirrors the barbs for the
/// southern hemisphere. Line work matches the Qt original: 25-px shaft, 10-px
/// full barbs, flags for 50 kt.
pub fn draw_barb(painter: &Painter, origin: Pos2, wdir: f64, wspd: f64, shemis: bool) {
    let color = barb_color(wspd);
    let stroke = Stroke::new(1.0, color);
    if !wspd.is_finite() || !wdir.is_finite() {
        return;
    }
    let mut spd = ((wspd / 5.0).round() * 5.0) as i64;
    if spd <= 0 {
        painter.circle_stroke(origin, 3.0, stroke);
        return;
    }
    // Build the path in barb-local coordinates (shaft along +x), then rotate
    // by (wdir - 90) degrees, matching QPainter::rotate in y-down space.
    let side = if shemis { -1.0 } else { 1.0 };
    let mut segments: Vec<(Vec2, Vec2)> = Vec::new();
    let mut pen = Vec2::new(0.0, 0.0);
    let line_to = |segments: &mut Vec<(Vec2, Vec2)>, pen: &mut Vec2, to: Vec2| {
        segments.push((*pen, to));
        *pen = to;
    };
    line_to(&mut segments, &mut pen, Vec2::new(25.0, 0.0));
    while spd >= 50 {
        let p = pen;
        line_to(&mut segments, &mut pen, Vec2::new(p.x, p.y + side * 10.0));
        let q = pen;
        line_to(&mut segments, &mut pen, Vec2::new(q.x - 4.0, p.y));
        pen = Vec2::new(q.x - 6.0, p.y);
        spd -= 50;
    }
    while spd >= 10 {
        let p = pen;
        line_to(&mut segments, &mut pen, Vec2::new(p.x, p.y + side * 10.0));
        pen = Vec2::new(p.x - 4.0, p.y);
        spd -= 10;
    }
    while spd >= 5 {
        let p = pen;
        line_to(&mut segments, &mut pen, Vec2::new(p.x, p.y + side * 5.0));
        pen = Vec2::new(p.x - 4.0, p.y);
        spd -= 5;
    }

    let ang = (wdir - 90.0).to_radians() as f32;
    let (sin, cos) = ang.sin_cos();
    let rot = |v: Vec2| Vec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos);
    for (a, b) in segments {
        painter.line_segment([origin + rot(a), origin + rot(b)], stroke);
    }
}
