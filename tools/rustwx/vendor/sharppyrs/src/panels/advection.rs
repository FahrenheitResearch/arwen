//! "Inf. Temp. Adv. (C/hr)" strip: port of `sharppy/viz/advection.py`
//! (`backgroundAdvection` + `plotAdvection`) with the SHARPpy-Reimagined
//! `ADV_TITLE_MAX_PT = 9` font cap from `sharpmod/render.py`. The per-layer
//! values come straight from [`DerivedParams::temp_adv`] /
//! [`DerivedParams::temp_adv_bounds`]; nothing is recomputed here.

use egui::{Align2, Color32, Painter, Pos2, Rect, Shape, Stroke};

use crate::Profile;
use crate::derived::DerivedParams;
use crate::skewt::SkewTStyle;
use crate::utils::{float2str, qc};

/// Qt point -> px at the standard 96-dpi factor.
const PT: f64 = 4.0 / 3.0;

/// Draw this panel into `rect`.
pub fn draw(painter: &Painter, rect: Rect, prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) {
    let _ = prof;
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
    // adv_max = 13, adv_min = 0 (the original resets adv_min to 0).

    let pt = |x: f64, y: f64| Pos2::new(rect.min.x + x as f32, rect.min.y + y as f32);
    let pres_to_pix = |pres: f64| bry - ((log_pmax - pres.ln()) / (log_pmax - log_pmin)) * bry;
    // adv_to_pix with rpad = 0 reduces to brx/2 + a*brx/26.
    let adv_to_pix = |a: f64| {
        let half = brx / 2.0;
        if a == 0.0 {
            half
        } else {
            brx * a / 26.0 + half
        }
    };

    // Label/title font: width * 0.12 + 3 pt, capped at 9 pt (Reimagined).
    let label_pt = ((w * 0.12).round() + 3.0).min(9.0);
    let label_font = style.regular_font((label_pt * PT) as f32);

    // Frame border.
    let border = Stroke::new(2.0, style.fg_color);
    p.line_segment([pt(0.0, 0.0), pt(brx, 0.0)], border);
    p.line_segment([pt(brx, 0.0), pt(brx, bry)], border);
    p.line_segment([pt(brx, bry), pt(0.0, bry)], border);
    p.line_segment([pt(0.0, bry), pt(0.0, 0.0)], border);

    // Dashed zero line.
    let x0 = adv_to_pix(0.0);
    p.extend(Shape::dashed_line(
        &[pt(x0, pres_to_pix(pmax)), pt(x0, pres_to_pix(pmin))],
        Stroke::new(1.0, style.fg_color),
        4.0,
        2.0,
    ));

    // Title, word-wrapped over two lines like the Qt original.
    let r = p.text(
        pt(2.0, 2.0),
        Align2::LEFT_TOP,
        "Inf. Temp. Adv.",
        label_font.clone(),
        style.fg_color,
    );
    p.text(
        pt(2.0, 2.0 + r.height() as f64),
        Align2::LEFT_TOP,
        "(C/hr)",
        label_font.clone(),
        style.fg_color,
    );

    // Layer boxes: red for warm advection, blue for cold, with value labels.
    let n = dv.temp_adv.len().min(dv.temp_adv_bounds.len());
    for i in 0..n {
        let adv = dv.temp_adv[i];
        let (pbot, ptop) = dv.temp_adv_bounds[i];
        if !qc(adv) || !qc(pbot) || !qc(ptop) {
            continue;
        }
        let pix_ptop = pres_to_pix(ptop);
        let pix_pbot = pres_to_pix(pbot);
        let pix_adv = adv_to_pix(adv);
        let (color, label_x) = if adv > 0.0 {
            (Color32::from_rgb(0xFF, 0x00, 0x00), adv_to_pix(8.0) - 5.0)
        } else if adv < 0.0 {
            (Color32::from_rgb(0x33, 0x99, 0xCC), adv_to_pix(-8.0))
        } else {
            (style.fg_color, adv_to_pix(8.0) - 5.0)
        };
        // Value label centered on a 5x8 slot at the layer midpoint
        // (TextDontClip | AlignCenter in the original).
        p.text(
            pt(label_x + 2.5, (pix_ptop + pix_pbot) / 2.0 + 4.0),
            Align2::CENTER_CENTER,
            float2str(adv, 1),
            label_font.clone(),
            color,
        );
        let stroke = Stroke::new(1.0, color);
        p.line_segment([pt(pix_adv, pix_ptop), pt(pix_adv, pix_pbot)], stroke);
        p.line_segment([pt(x0, pix_pbot), pt(pix_adv, pix_pbot)], stroke);
        p.line_segment([pt(pix_adv, pix_ptop), pt(x0, pix_ptop)], stroke);
        p.line_segment([pt(x0, pix_ptop), pt(x0, pix_pbot)], stroke);
    }
}
