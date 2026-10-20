//! "SR Wind v. Height" inset: port of `sharppy/viz/srwinds.py`
//! (`backgroundWinds` + `plotWinds`, right mover) with the SHARPpy-Reimagined
//! layout fixes from `sharpmod/viz/inset_layout.py`: title drawn in the
//! Theta-E-style box, height labels left-anchored inside the frame, and the
//! "Classic Supercell" annotation centered over the 40-70 kt band and clamped
//! inside the widget.
//!
//! Storm motion is `prof.srwind` (right mover). The red trace is the
//! storm-relative wind speed interpolated to 10 m steps; the 0-2 / 4-6 /
//! 9-11 km mean-SR-wind bars use `sharprs::winds::sr_wind` over the same
//! layers as SHARPpy's convective profile.

use egui::{Align2, Color32, Painter, Pos2, Rect, Shape, Stroke};

use crate::Profile;
use crate::derived::DerivedParams;
use crate::skewt::SkewTStyle;
use crate::utils::{int2str, qc};

/// Qt point -> px at the standard 96-dpi factor.
const PT: f64 = 4.0 / 3.0;

/// Linear interpolation with clamped ends (np.interp semantics; `xs` ascending).
fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    let last = xs.len() - 1;
    if x >= xs[last] {
        return ys[last];
    }
    let mut lo = 0;
    let mut hi = last;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if xs[hi] == xs[lo] {
        ys[lo]
    } else {
        ys[lo] + (x - xs[lo]) / (xs[hi] - xs[lo]) * (ys[hi] - ys[lo])
    }
}

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
    let (hmax, smax) = (16.0f64, 80.0f64); // km / kts

    let pt = |x: f64, y: f64| Pos2::new(rect.min.x + x as f32, rect.min.y + y as f32);
    let hgt_to_pix = |hkm: f64| (bry - 2.0) - (hkm / hmax) * bry;
    // Faithful quirk of the original: the speed axis is anchored on bry.
    let speed_to_pix = |s: f64| bry - ((smax - s) / smax) * brx;

    let label_pt = (h * 0.0512).round();
    let label_font = style.regular_font((label_pt * PT) as f32);
    let fg = style.fg_color;
    let clsc = Color32::from_rgb(0xB1, 0x01, 0x9A);

    // Frame border.
    let border = Stroke::new(2.0, fg);
    p.line_segment([pt(0.0, 0.0), pt(brx, 0.0)], border);
    p.line_segment([pt(brx, 0.0), pt(brx, bry)], border);
    p.line_segment([pt(brx, bry), pt(0.0, bry)], border);
    p.line_segment([pt(0.0, bry), pt(0.0, 0.0)], border);

    // Title: three lines centered on the Theta-E-style (35, 15, 50, 50) box.
    let lh = p
        .layout_no_wrap("SR Wind".to_string(), label_font.clone(), fg)
        .size()
        .y as f64;
    p.text(
        pt(60.0, 40.0 - lh),
        Align2::CENTER_CENTER,
        "SR Wind",
        label_font.clone(),
        fg,
    );
    p.text(
        pt(60.0, 40.0),
        Align2::CENTER_CENTER,
        "v.",
        label_font.clone(),
        fg,
    );
    p.text(
        pt(60.0, 40.0 + lh),
        Align2::CENTER_CENTER,
        "Height",
        label_font.clone(),
        fg,
    );

    // 15 kt dashed reference line.
    let x15 = speed_to_pix(15.0);
    p.extend(Shape::dashed_line(
        &[pt(x15, bry), pt(x15, 0.0)],
        Stroke::new(1.0, fg),
        4.0,
        2.0,
    ));

    // Classic Supercell band: dashed 40/70 kt lines over 8-16 km + label.
    let lower = hgt_to_pix(8.0);
    let upper = hgt_to_pix(16.0);
    let c1 = speed_to_pix(40.0);
    let c2 = speed_to_pix(70.0);
    let clsc_stroke = Stroke::new(1.0, clsc);
    p.extend(Shape::dashed_line(
        &[pt(c1, lower), pt(c1, upper)],
        clsc_stroke,
        4.0,
        2.0,
    ));
    p.extend(Shape::dashed_line(
        &[pt(c2, lower), pt(c2, upper)],
        clsc_stroke,
        4.0,
        2.0,
    ));
    let tw = 74.0;
    let hi = brx - tw - 1.0;
    let tx = if hi > 1.0 {
        ((c1 + c2) / 2.0 - tw / 2.0).clamp(1.0, hi)
    } else {
        1.0
    };
    // Two lines vertically centered in the (tx, 2, 74, 40) box.
    p.text(
        pt(tx + tw / 2.0, 22.0 - lh / 2.0),
        Align2::CENTER_CENTER,
        "Classic",
        label_font.clone(),
        clsc,
    );
    p.text(
        pt(tx + tw / 2.0, 22.0 + lh / 2.0),
        Align2::CENTER_CENTER,
        "Supercell",
        label_font.clone(),
        clsc,
    );

    // Height ticks (km) + labels (patched: left-anchored inside the frame).
    let tick = Stroke::new(1.0, fg);
    for hkm in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0] {
        let y1 = hgt_to_pix(hkm);
        p.line_segment([pt(0.0, y1), pt(5.0, y1)], tick);
        p.line_segment([pt(brx - 5.0, y1), pt(brx, y1)], tick);
        p.text(
            pt(7.0, y1),
            Align2::LEFT_CENTER,
            int2str(hkm),
            label_font.clone(),
            fg,
        );
    }
    // Speed ticks every 10 kt along the top and bottom edges.
    let mut s = 0.0f64;
    while s < 100.0 {
        let x1 = speed_to_pix(s);
        p.line_segment([pt(x1, 0.0), pt(x1, 5.0)], tick);
        p.line_segment([pt(x1, bry - 5.0), pt(x1, bry)], tick);
        s += 10.0;
    }

    // ------------------------------------------------------------------
    // Data
    // ------------------------------------------------------------------
    let (smu, smv) = (prof.srwind.0, prof.srwind.1);
    if !qc(smu) || !qc(smv) {
        return;
    }
    let inner = &prof.inner;

    // SR wind speed trace, interpolated to 10 m steps (like the original).
    let mut hs: Vec<f64> = Vec::new();
    let mut sru: Vec<f64> = Vec::new();
    let mut srv: Vec<f64> = Vec::new();
    for i in 0..inner.pres.len() {
        if qc(inner.u[i]) && qc(inner.v[i]) && inner.hght[i].is_finite() {
            hs.push(inner.hght[i]);
            sru.push(inner.u[i] - smu);
            srv.push(inner.v[i] - smv);
        }
    }
    if hs.len() >= 2 {
        let h0 = if inner.hght[inner.sfc].is_finite() {
            inner.hght[inner.sfc]
        } else {
            hs[0]
        };
        let hend = hs[hs.len() - 1].min(hmax * 1000.0);
        let mut pts: Vec<Pos2> = Vec::new();
        let mut hh = h0;
        while hh < hend {
            let su = interp(&hs, &sru, hh);
            let sv = interp(&hs, &srv, hh);
            pts.push(pt(
                speed_to_pix(su.hypot(sv)),
                hgt_to_pix((hh - h0) / 1000.0),
            ));
            hh += 10.0;
        }
        if pts.len() >= 2 {
            p.add(Shape::line(
                pts,
                Stroke::new(1.0, Color32::from_rgb(0xFF, 0x00, 0x00)),
            ));
        }
    }

    // Layer-mean SR wind bars (0-2 / 4-6 / 9-11 km).
    let pres_at_agl = |m: f64| inner.pres_at_height(inner.to_msl(m));
    let bar = |pbot: f64, ptop: f64, h1: f64, h2: f64, color: Color32| {
        if !qc(pbot) || !qc(ptop) {
            return;
        }
        if let Ok((u, v)) = sharprs::winds::sr_wind(inner, pbot, ptop, smu, smv, -1.0) {
            let spd = u.hypot(v);
            if qc(spd) {
                let x = speed_to_pix(spd);
                p.line_segment(
                    [pt(x, hgt_to_pix(h1)), pt(x, hgt_to_pix(h2))],
                    Stroke::new(2.0, color),
                );
            }
        }
    };
    bar(
        inner.pres[inner.sfc],
        pres_at_agl(2000.0),
        0.0,
        2.0,
        Color32::from_rgb(0x8B, 0x00, 0x00),
    );
    bar(
        pres_at_agl(4000.0),
        pres_at_agl(6000.0),
        4.0,
        6.0,
        Color32::from_rgb(0x64, 0x95, 0xED),
    );
    bar(
        pres_at_agl(9000.0),
        pres_at_agl(11000.0),
        9.0,
        11.0,
        Color32::from_rgb(0x94, 0x00, 0xD3),
    );
}
