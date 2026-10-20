//! "Psbl Haz. Type" box: the vendored watch-box look (`sharppy/viz/watch.py`:
//! centered white title, separator, big centered hazard word) driven by the
//! vendored `sharppy.sharptab.watch_type.possible_watch` decision cascade —
//! the logic behind the reference render's "PDS TOR" readout. Labels and
//! colors match `watch.py` (`PDS TOR` #ff00ff, `TOR`/`MRGL TOR` #ff0000,
//! `SVR` #ffff00, `MRGL SVR` #0099cc); only the most severe type is shown.
//! The FLASH FLOOD / BLIZZARD / heat categories need climatology or precip
//! type inputs outside this crate's scope and are omitted.

use egui::{Align2, Color32, Painter, Pos2, Rect, Stroke};

use crate::Profile;
use crate::derived::DerivedParams;
use crate::skewt::SkewTStyle;
use crate::utils::qc;

/// Qt point -> px at the standard 96-dpi factor.
const PT: f64 = 4.0 / 3.0;

/// Port of `watch_type.possible_watch` (right-mover branch): returns the most
/// severe watch type and its display color.
fn classify(prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) -> (&'static str, Color32) {
    let none = ("NONE", style.fg_color);
    let inner = &prof.inner;

    // lr1 = lapse_rate(0, 1000 m AGL) on virtual temperature, like the vendored params.
    let lr1 = sharprs::params::indices::lapse_rate(inner, 0.0, 1000.0, false).unwrap_or(f64::NAN);
    let mut stp_eff = dv.stp_cin;
    let mut stp_fixed = dv.stp_fixed;
    // right_srw_4_6km: SR wind over the 4-6 km AGL layer, right mover.
    let p4 = inner.pres_at_height(inner.to_msl(4000.0));
    let p6 = inner.pres_at_height(inner.to_msl(6000.0));
    let srw_4_6km = sharprs::winds::sr_wind(inner, p4, p6, prof.srwind.0, prof.srwind.1, -1.0)
        .map(|(u, v)| u.hypot(v))
        .unwrap_or(f64::NAN);
    let mut esrh = prof.right_esrh;
    let mut srh1km = dv.srh1km;
    if prof.latitude() < 0.0 {
        stp_eff = -stp_eff;
        stp_fixed = -stp_fixed;
        esrh = -esrh;
        srh1km = -srh1km;
    }
    let sfc_8km_shear = if qc(dv.sfc_8km_shear.0) {
        dv.sfc_8km_shear.0.hypot(dv.sfc_8km_shear.1)
    } else {
        f64::NAN
    };
    let sfc_lcl = prof.sfcpcl.lclhght;
    let ml_lcl = prof.mlpcl.lclhght;
    let ml_cin = prof.mlpcl.bminus;
    let mu_cin = prof.mupcl.bminus;
    // Surface-based effective layer. The Python compares `ebotm == 0`
    // exactly; our interpolation round-trips through log-pressure and can
    // return ±1e-13 m for a surface-based layer, which silently failed every
    // TOR gate (soundings read SVR instead of PDS TOR/TOR). Compare with a
    // millimeter tolerance instead — NaN (no effective layer) stays false.
    let sfc_based_eff = prof.ebotm.abs() < 1e-3;
    let scp = dv.right_scp;

    let pds_tor = ("PDS TOR", Color32::from_rgb(0xFF, 0x00, 0xFF));
    let tor = ("TOR", Color32::from_rgb(0xFF, 0x00, 0x00));
    let mrgl_tor = ("MRGL TOR", Color32::from_rgb(0xFF, 0x00, 0x00));
    let svr = ("SVR", Color32::from_rgb(0xFF, 0xFF, 0x00));
    let mrgl_svr = ("MRGL SVR", Color32::from_rgb(0x00, 0x99, 0xCC));

    // TOR cascade (NaN comparisons are false, matching masked semantics).
    if stp_eff >= 3.0
        && stp_fixed >= 3.0
        && srh1km >= 200.0
        && esrh >= 200.0
        && srw_4_6km >= 15.0
        && sfc_8km_shear > 45.0
        && sfc_lcl < 1000.0
        && ml_lcl < 1200.0
        && lr1 >= 5.0
        && ml_cin > -50.0
        && sfc_based_eff
    {
        return pds_tor;
    }
    if (stp_eff >= 3.0 || stp_fixed >= 4.0) && ml_cin > -125.0 && sfc_based_eff {
        return tor;
    }
    if (stp_eff >= 1.0 || stp_fixed >= 1.0)
        && (srw_4_6km >= 15.0 || sfc_8km_shear >= 40.0)
        && ml_cin > -50.0
        && sfc_based_eff
    {
        return tor;
    }
    if (stp_eff >= 1.0 || stp_fixed >= 1.0)
        && (dv.low_rh + dv.mid_rh) / 2.0 >= 60.0
        && lr1 >= 5.0
        && ml_cin > -50.0
        && sfc_based_eff
    {
        return tor;
    }
    if (stp_eff >= 1.0 || stp_fixed >= 1.0) && ml_cin > -150.0 && sfc_based_eff {
        return mrgl_tor;
    }
    // Vendored operator-precedence quirk kept: the `or` binds the whole
    // left clause against the (cin && sfc-based) right clause.
    if (stp_eff >= 0.5 && esrh >= 150.0)
        || (stp_fixed >= 0.5 && srh1km >= 150.0 && ml_cin > -50.0 && sfc_based_eff)
    {
        return mrgl_tor;
    }

    // SVR cascade.
    if (stp_fixed >= 1.0 || scp >= 4.0 || stp_eff >= 1.0) && mu_cin >= -50.0 {
        return svr;
    }
    if scp >= 2.0 && (dv.ship >= 1.0 || dv.dcape >= 750.0) && mu_cin >= -50.0 {
        return svr;
    }
    if dv.sig_severe >= 30000.0 && dv.mmp >= 0.6 && mu_cin >= -50.0 {
        return svr;
    }
    if mu_cin >= -75.0 && (dv.wndg >= 0.5 || dv.ship >= 0.5 || scp >= 0.5) {
        return mrgl_svr;
    }
    none
}

/// Draw this panel into `rect`.
pub fn draw(painter: &Painter, rect: Rect, prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) {
    let p = painter.with_clip_rect(rect);
    p.rect_filled(rect, 0.0, style.bg_color);

    // Geometry (watch.py initUI): lpad = rpad = tpad = 0, bpad = 20.
    let w = rect.width() as f64;
    let h = rect.height() as f64;
    let bpad = 20.0;
    let brx = w;
    let bry = h - bpad;
    let pad = bry / 100.0;

    let pt = |x: f64, y: f64| Pos2::new(rect.min.x + x as f32, rect.min.y + y as f32);

    // Fonts (font_ratio = 0.0512, like the vendored watch box).
    let title_pt = (h * 0.0512).round() + 5.0;
    let plot_pt = (h * 0.0512).round() + 4.0;
    let title_font = style.regular_font((title_pt * PT) as f32);
    let plot_font = style.regular_font((plot_pt * PT) as f32);
    let fg = style.fg_color;

    // Frame border.
    let border = Stroke::new(2.0, fg);
    p.line_segment([pt(0.0, 0.0), pt(brx, 0.0)], border);
    p.line_segment([pt(brx, 0.0), pt(brx, bry)], border);
    p.line_segment([pt(brx, bry), pt(0.0, bry)], border);
    p.line_segment([pt(0.0, bry), pt(0.0, 0.0)], border);

    // Title bar + separator.
    let title_h = p
        .layout_no_wrap("Psbl Haz. Type".to_string(), title_font.clone(), fg)
        .size()
        .y as f64;
    p.text(
        pt(brx / 2.0, pad * 4.0 + title_h / 2.0),
        Align2::CENTER_CENTER,
        "Psbl Haz. Type",
        title_font,
        fg,
    );
    let sep_y = pad * 4.0 + title_h + 3.0;
    p.line_segment([pt(0.0, sep_y), pt(brx, sep_y)], Stroke::new(1.0, fg));

    // The big hazard word, colored per hazard.
    let (label, color) = classify(prof, dv, style);
    p.text(
        pt(brx / 2.0, bry / 2.0 + title_h / 2.0),
        Align2::CENTER_CENTER,
        label,
        plot_font,
        color,
    );
}
