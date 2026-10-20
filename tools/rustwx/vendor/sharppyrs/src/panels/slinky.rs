//! "Storm Slinky" inset: port of `sharppy/viz/slinky.py` (`backgroundSlinky`
//! + `plotSlinky`) with the SHARPpy-Reimagined title fit from
//! `sharpmod/render.py::_install_slinky_title_fit` (title anchored just above
//! the bottom border at its full font height).
//!
//! The trajectory itself comes from [`DerivedParams::slinky_traj`] /
//! [`DerivedParams::slinky_tilt`] (the `params.parcelTraj` port) and is never
//! recomputed here. `slinky_traj` only carries the horizontal `(x, y)`
//! displacements, so the per-circle height AGL used for coloring is
//! reconstructed from the parcel's LFC/EL heights assuming the trajectory's
//! constant time step (25 s), initial 5 m/s nudge, and a constant mean
//! buoyancy acceleration — a close approximation of the original z sequence.

use egui::{Align2, Color32, Painter, Pos2, Rect, Stroke};

use crate::Profile;
use crate::derived::DerivedParams;
use crate::skewt::SkewTStyle;
use crate::utils::{int2str, qc};

/// Qt point -> px at the standard 96-dpi factor.
const PT: f64 = 4.0 / 3.0;

/// Draw this panel into `rect`.
pub fn draw(painter: &Painter, rect: Rect, prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) {
    let p = painter.with_clip_rect(rect);
    p.rect_filled(rect, 0.0, style.bg_color);

    // Geometry (initUI): lpad = 5, rpad = tpad = 0, bpad = 20.
    let w = rect.width() as f64;
    let h = rect.height() as f64;
    let lpad = 5.0;
    let bpad = 20.0;
    let brx = w;
    let bry = h - bpad;
    let (centerx, centery) = (w / 2.0, bry / 2.0);
    let mag = 7000.0 * 1.7;
    let scale = brx / mag;

    let pt = |x: f64, y: f64| Pos2::new(rect.min.x + x as f32, rect.min.y + y as f32);
    let xy_to_pix = |x: f64, y: f64| pt(centerx + x * scale, centery - y * scale);

    let title_pt = (h * 0.0512).round() + 2.0;
    let title_font = style.regular_font((title_pt * PT) as f32);

    // X/Y axes.
    let axes = Stroke::new(2.0, Color32::from_rgb(0x00, 0x33, 0x66));
    p.line_segment([pt(centerx, 0.0), pt(centerx, bry)], axes);
    p.line_segment([pt(0.0, centery), pt(brx, centery)], axes);

    // Frame border.
    let border = Stroke::new(2.0, style.fg_color);
    p.line_segment([pt(0.0, 0.0), pt(brx, 0.0)], border);
    p.line_segment([pt(brx, 0.0), pt(brx, bry)], border);
    p.line_segment([pt(brx, bry), pt(0.0, bry)], border);
    p.line_segment([pt(0.0, bry), pt(0.0, 0.0)], border);

    // Title, full font height above the bottom border (Reimagined title fit).
    p.text(
        pt(lpad, bry - 2.0),
        Align2::LEFT_BOTTOM,
        "Storm Slinky",
        title_font.clone(),
        style.fg_color,
    );

    // Trajectory circles, colored by height AGL like the hodograph.
    let traj = &dv.slinky_traj;
    let n = traj.len();
    if n > 0 {
        let pcl = &prof.mupcl;
        let has_el = pcl.bplus > 1e-3 && qc(pcl.elhght);
        // Reconstruct the trajectory heights (see module docs).
        let z0 = if qc(pcl.lfchght) { pcl.lfchght } else { 0.0 };
        let zend = if qc(pcl.elhght) {
            pcl.elhght
        } else {
            prof.inner
                .hght
                .iter()
                .rev()
                .cloned()
                .find(|z| z.is_finite())
                .map(|z| prof.inner.to_agl(z))
                .unwrap_or(z0)
        };
        let t_total = 25.0 * n.saturating_sub(1) as f64;
        let accel = if t_total > 0.0 {
            2.0 * (zend - z0 - 5.0 * t_total) / (t_total * t_total)
        } else {
            0.0
        };
        for (i, &(x, y)) in traj.iter().enumerate() {
            if !qc(x) || !qc(y) {
                continue;
            }
            let t = 25.0 * i as f64;
            let z = z0 + 5.0 * t + 0.5 * accel * t * t;
            let color = if has_el && i == n - 1 {
                Color32::from_rgb(0xFF, 0x00, 0xFF)
            } else if z < 3000.0 {
                Color32::from_rgb(0xFF, 0x00, 0x00)
            } else if z < 6000.0 {
                Color32::from_rgb(0x00, 0xFF, 0x00)
            } else if z < 9000.0 {
                Color32::from_rgb(0xFF, 0xFF, 0x00)
            } else if z < 12000.0 {
                Color32::from_rgb(0x00, 0xFF, 0xFF)
            } else {
                continue;
            };
            p.circle_stroke(xy_to_pix(x, y), 5.0, Stroke::new(1.0, color));
        }
    }

    // Storm motion vector (right mover), scaled to 3000 m in slinky space.
    let (smu, smv) = (prof.srwind.0, prof.srwind.1);
    if qc(smu) && qc(smv) {
        let m = smu.hypot(smv);
        if m > 0.0 {
            let (u, v) = (smu / m * 3000.0, smv / m * 3000.0);
            p.line_segment(
                [xy_to_pix(u, v), xy_to_pix(0.0, 0.0)],
                Stroke::new(2.0, style.fg_color),
            );
        }
    }

    // Updraft tilt readout, top right.
    p.text(
        pt(brx - 4.0, 2.0),
        Align2::RIGHT_TOP,
        format!("{} deg", int2str(dv.slinky_tilt)),
        title_font,
        style.fg_color,
    );
}
