//! "Wind Speed (knots)" vs log-p strip: port of `sharppy/viz/speed.py`
//! (`backgroundSpeed` + `plotSpeed`) with the SHARPpy-Reimagined overrides
//! from `sharpmod/render.py`: title/axis fonts capped at 9 pt
//! (`SPEED_TITLE_MAX_PT` / `SPEED_LABEL_MAX_PT`) and the SFC-500 m layer of
//! the profile colored magenta like the hodograph split
//! (`_install_speed_0500`, `HODO_0_500_COLOR`).

use egui::{Align2, Color32, Painter, Pos2, Rect, Shape, Stroke};

use crate::Profile;
use crate::derived::DerivedParams;
use crate::skewt::SkewTStyle;
use crate::utils::{int2str, qc};

/// Qt point -> px at the standard 96-dpi factor.
const PT: f64 = 4.0 / 3.0;

/// Draw this panel into `rect`.
pub fn draw(painter: &Painter, rect: Rect, prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) {
    let _ = dv;
    let p = painter.with_clip_rect(rect);
    p.rect_filled(rect, 0.0, style.bg_color);

    // Geometry (initUI): lpad = rpad = tpad = 0, bpad = 20.
    let w = rect.width() as f64;
    let h = rect.height() as f64;
    let bpad = 20.0;
    let brx = w;
    let bry = h - bpad;
    let (pmax, pmin) = (1050.0f64, 100.0f64);
    let (log_pmax, log_pmin) = (pmax.ln(), pmin.ln());
    let (smax, smin) = (140.0f64, 0.0f64); // knots
    let delta = 20.0f64;

    let pt = |x: f64, y: f64| Pos2::new(rect.min.x + x as f32, rect.min.y + y as f32);
    let pres_to_pix = |pres: f64| bry - ((log_pmax - pres.ln()) / (log_pmax - log_pmin)) * bry;
    let speed_to_pix = |s: f64| brx - ((smax - s) / (smax - smin)) * brx;

    // Fonts sized from the strip width (font_ratio = 0.12), capped at 9 pt by
    // the SHARPpy-Reimagined render overrides.
    let title_pt = (w * 0.12).round().min(9.0) + 1.0;
    let label_pt = ((w * 0.12).round() + 2.0).min(9.0);
    let title_font = style.regular_font((title_pt * PT) as f32);
    let label_font = style.regular_font((label_pt * PT) as f32);

    // Frame border.
    let border = Stroke::new(2.0, style.fg_color);
    p.line_segment([pt(0.0, 0.0), pt(brx, 0.0)], border);
    p.line_segment([pt(brx, 0.0), pt(brx, bry)], border);
    p.line_segment([pt(brx, bry), pt(0.0, bry)], border);
    p.line_segment([pt(0.0, bry), pt(0.0, 0.0)], border);

    // Dashed isotachs every 20 kt; labels at 40/80/120 in the bottom band.
    let isotach = Stroke::new(1.0, Color32::from_rgb(0x9D, 0x57, 0x36));
    let mut s = smin;
    while s < smax {
        let x1 = speed_to_pix(s);
        p.extend(Shape::dashed_line(
            &[pt(x1, bry), pt(x1, 0.0)],
            isotach,
            4.0,
            2.0,
        ));
        if (s as i64) % (delta as i64 * 2) == 0 && s > 0.0 {
            // Patched draw_speed: label centered on the isotach in a slot of
            // height bpad - 4 starting at bry + 1.
            p.text(
                pt(x1, bry + 1.0 + (bpad - 4.0) / 2.0),
                Align2::CENTER_CENTER,
                int2str(s),
                label_font.clone(),
                style.fg_color,
            );
        }
        s += delta;
    }

    // Title, word-wrapped over two lines like the Qt original.
    let r = p.text(
        pt(2.0, 2.0),
        Align2::LEFT_TOP,
        "Wind Speed",
        title_font.clone(),
        style.fg_color,
    );
    p.text(
        pt(2.0, 2.0 + r.height() as f64),
        Align2::LEFT_TOP,
        "(knots)",
        title_font,
        style.fg_color,
    );

    // Wind speed bars per level, colored by height AGL with the hodograph
    // colors (plus the Reimagined magenta SFC-500 m band).
    let inner = &prof.inner;
    for i in 0..inner.pres.len() {
        let (u, v, hg, pr) = (inner.u[i], inner.v[i], inner.hght[i], inner.pres[i]);
        if !qc(u) || !qc(v) || !qc(hg) || !qc(pr) {
            continue;
        }
        let spd = u.hypot(v);
        let agl = inner.to_agl(hg);
        let color = if agl < 500.0 {
            Color32::from_rgb(0xFF, 0x00, 0xFF)
        } else if agl < 3000.0 {
            Color32::from_rgb(0xFF, 0x00, 0x00)
        } else if agl < 6000.0 {
            Color32::from_rgb(0x00, 0xFF, 0x00)
        } else if agl < 9000.0 {
            Color32::from_rgb(0xFF, 0xFF, 0x00)
        } else {
            Color32::from_rgb(0x00, 0xFF, 0xFF)
        };
        let y1 = pres_to_pix(pr);
        p.line_segment(
            [pt(0.0, y1), pt(speed_to_pix(spd), y1)],
            Stroke::new(2.0, color),
        );
    }
}
