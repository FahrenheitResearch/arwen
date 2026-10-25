//! The SPC-style hodograph panel: a faithful egui port of
//! `sharppy.viz.hodo` (`backgroundHodo.plotBackground` + `plotHodo.plotData`,
//! SHARPpy 1.4.0a5) plus the SHARPpy-Reimagined overlay passes:
//! the 200-kt zoom (`render._install_hodo_zoom`, `HODO_ZOOM_KTS`), the
//! 0-500 m magenta trace band (`render._install_hodo_0500`) and the top-left
//! locator inset (`sharpmod.viz.hodo_locator`, drawn here from an embedded
//! Natural Earth basemap instead of the live TIGERweb county query).
//! Interactivity (dragging the trace / storm-motion vectors, cursor readouts,
//! boundary cursor) is intentionally not ported.

use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Shape, Stroke, StrokeKind, Vec2};

use sharprs::profile::{comp2vec, vec2comp};

use crate::Profile;
use crate::derived::DerivedParams;
use crate::skewt::SkewTStyle;
use crate::utils::{int2str, qc};

use super::basemap;

// SPC default colors of the original (config defaults hardcoded, like the
// Python widget does for everything the preferences dialog cannot change).
const ISOTACH_COLOR: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);
/// Height-band trace colors: 0-500 m (SHARPpy-Reimagined `HODO_0_500_COLOR`),
/// then the config defaults for 0-3 / 3-6 / 6-9 / 9-12 km.
const TRACE_COLORS: [Color32; 5] = [
    Color32::from_rgb(0xFF, 0x00, 0xFF),
    Color32::from_rgb(0xFF, 0x00, 0x00),
    Color32::from_rgb(0x00, 0xFF, 0x00),
    Color32::from_rgb(0xFF, 0xFF, 0x00),
    Color32::from_rgb(0x00, 0xFF, 0xFF),
];
const EFF_INFLOW_COLOR: Color32 = Color32::from_rgb(0x00, 0xFF, 0xFF);
const CRIT_TEXT_COLOR: Color32 = Color32::from_rgb(0x00, 0xFF, 0xFF);
const CRIT_LINE_COLOR: Color32 = Color32::from_rgb(0xFF, 0x00, 0xFF);
const CORFIDI_COLOR: Color32 = Color32::from_rgb(0x00, 0xBF, 0xFF);
const MEAN_LCL_EL_COLOR: Color32 = Color32::from_rgb(0xB8, 0x86, 0x0B);
// Locator inset palette (sharpmod.viz.hodo_locator module constants).
const MAP_FILL: Color32 = Color32::from_rgb(0x05, 0x09, 0x0B);
const MAP_POINT_COLOR: Color32 = Color32::from_rgb(0xFF, 0xDA, 0x00);

/// Geometry + transforms (port of `backgroundHodo.initUI` with the
/// SHARPpy-Reimagined zoom, widened ~12% per field feedback).
/// Default hodograph window (kts across); SHARPpy-Reimagined uses 200,
/// widened 25% per field feedback. Scroll-zoomable in `SoundingView`.
pub(crate) const DEFAULT_ZOOM_KTS: f64 = 250.0;

struct Geom {
    rect: Rect,
    wid: f64,
    hgt: f64,
    centerx: f64,
    centery: f64,
    /// Pixels per knot.
    scale: f64,
}

impl Geom {
    fn with_zoom(rect: Rect, zoom_kts: f64) -> Geom {
        let wid = rect.width() as f64;
        let hgt = rect.height() as f64;
        let hodomag = zoom_kts.clamp(80.0, 500.0);
        Geom {
            rect,
            wid,
            hgt,
            centerx: wid / 2.0,
            centery: hgt / 2.0,
            scale: wid / hodomag,
        }
    }

    /// (u, v) kts -> widget-local px.
    fn uv_to_pix(&self, u: f64, v: f64) -> (f64, f64) {
        (self.centerx + u * self.scale, self.centery - v * self.scale)
    }

    /// Widget-local -> screen position.
    fn pt(&self, x: f64, y: f64) -> Pos2 {
        Pos2::new(self.rect.min.x + x as f32, self.rect.min.y + y as f32)
    }
}

/// Fonts sized from the widget height like the Qt original (pt -> px at the
/// standard 96-dpi factor 4/3). Hodo fsize = 7 (the > 75-dpi branch).
struct Fonts {
    label: FontId,
    critical: FontId,
    critical_height: f64,
}

impl Fonts {
    fn new(hgt: f64, style: &SkewTStyle) -> Fonts {
        const PT: f64 = 4.0 / 3.0;
        let fsize = 7.0;
        let label_pt = fsize + hgt * 0.0045;
        let critical_pt = fsize + 2.0 + hgt * 0.0045;
        Fonts {
            label: style.bold_font((label_pt * PT) as f32),
            critical: style.bold_font((critical_pt * PT) as f32),
            // xHeight() + 5 + hgt*0.0045, approximated like skewt::Fonts.
            critical_height: critical_pt * PT * 0.5 + 5.0 + hgt * 0.0045,
        }
    }
}

#[cfg(test)]
mod typography_tests {
    use super::*;
    use crate::skewt::SoundingFontPreset;

    #[test]
    fn shared_typography_reaches_hodograph_labels_without_geometry_scaling() {
        let base = Fonts::new(440.0, &SkewTStyle::default());
        let style = SkewTStyle::default()
            .with_font_preset(SoundingFontPreset::TechnicalMonospace)
            .with_text_scale(1.4);
        let scaled = Fonts::new(440.0, &style);

        assert!((scaled.label.size - base.label.size * 1.4).abs() < 0.001);
        assert_eq!(scaled.label.family, egui::FontFamily::Monospace);
        assert_eq!(scaled.critical_height, base.critical_height);
    }
}

/// Draw this panel into `rect`.
pub fn draw(painter: &Painter, rect: Rect, prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) {
    draw_zoomed(painter, rect, prof, dv, style, DEFAULT_ZOOM_KTS)
}

/// `draw` with an explicit hodograph window width in knots.
pub(crate) fn draw_zoomed(
    painter: &Painter,
    rect: Rect,
    prof: &Profile,
    dv: &DerivedParams,
    style: &SkewTStyle,
    zoom_kts: f64,
) {
    if rect.width() < 50.0 || rect.height() < 50.0 {
        return;
    }
    let p = painter.with_clip_rect(rect);
    let g = Geom::with_zoom(rect, zoom_kts);
    let fonts = Fonts::new(g.hgt, style);

    p.rect_filled(rect, 0.0, style.bg_color);
    draw_background(&p, &g, &fonts, style);
    draw_data(&p, &g, &fonts, prof, dv, style);
    // The locator map now lives in its own panel (see panels::locator);
    // keep the hodograph corner clear like upstream SHARPpy.
    let _ = draw_locator;
}

// ----------------------------------------------------------------------
// Background pass (port of backgroundHodo.plotBackground)
// ----------------------------------------------------------------------
fn draw_background(p: &Painter, g: &Geom, fonts: &Fonts, style: &SkewTStyle) {
    // Speed rings out to the bottom-right corner, every 10 kt.
    let max_uv = (g.centerx.hypot(g.centery) / g.scale) as i64;
    let mut spd = 10i64;
    while spd <= max_uv + 10 {
        draw_ring(p, g, fonts, style, spd as f64);
        spd += 10;
    }
    // Axes + frame (fg white, 2 px).
    let stroke = Stroke::new(2.0, style.fg_color);
    p.line_segment([g.pt(g.centerx, 0.0), g.pt(g.centerx, g.hgt)], stroke);
    p.line_segment([g.pt(0.0, g.centery), g.pt(g.wid, g.centery)], stroke);
    p.line_segment([g.pt(0.0, 0.0), g.pt(g.wid, 0.0)], stroke);
    p.line_segment([g.pt(g.wid, 0.0), g.pt(g.wid, g.hgt)], stroke);
    p.line_segment([g.pt(g.wid, g.hgt), g.pt(0.0, g.hgt)], stroke);
    p.line_segment([g.pt(0.0, g.hgt), g.pt(0.0, 0.0)], stroke);
}

/// Port of `backgroundHodo.draw_ring`: dashed isotach circle + the four
/// speed labels sitting on the axes.
fn draw_ring(p: &Painter, g: &Geom, fonts: &Fonts, style: &SkewTStyle, spd: f64) {
    let r = spd * g.scale;
    // Qt drawEllipse with a DashLine pen (dash 4, gap 2 at width 1).
    let n = (r as usize).clamp(64, 512);
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let a = std::f64::consts::TAU * i as f64 / n as f64;
        pts.push(g.pt(g.centerx + r * a.cos(), g.centery + r * a.sin()));
    }
    p.extend(Shape::dashed_line(
        &pts,
        Stroke::new(1.0, ISOTACH_COLOR),
        4.0,
        2.0,
    ));

    // Labels: AlignCenter in 15x15 rects offset 5 px from the axes
    // (centers at +/- ring radius, nudged +12.5 / +2.5 like the original).
    let text = int2str(spd);
    let centers = [
        (g.centerx + 12.5, g.centery - r + 2.5),
        (g.centerx + 12.5, g.centery + r + 2.5),
        (g.centerx - r + 2.5, g.centery + 12.5),
        (g.centerx + r + 2.5, g.centery + 12.5),
    ];
    for (x, y) in centers {
        p.text(
            g.pt(x, y),
            Align2::CENTER_CENTER,
            text.clone(),
            fonts.label.clone(),
            style.fg_color,
        );
    }
}

// ----------------------------------------------------------------------
// Data pass (port of plotHodo.plotData)
// ----------------------------------------------------------------------
fn draw_data(
    p: &Painter,
    g: &Geom,
    fonts: &Fonts,
    prof: &Profile,
    dv: &DerivedParams,
    style: &SkewTStyle,
) {
    // ONLY DRAW A HODOGRAPH IF THERE'S WIND DATA (prof.wdir.count() > 1).
    if !draw_hodo_trace(p, g, prof) {
        return;
    }
    draw_smv(p, g, fonts, prof, style);
    draw_corfidi(p, g, fonts, dv);
    draw_lcl_to_el_mw(p, g, fonts, dv);
    draw_critical_angle(p, g, fonts, prof, dv, style);
}

/// Linear interpolation of `f` at `zt` over ascending `z`
/// (port of `interp.generic_interp_hght` / np.interp, clamped at the ends).
fn interp_z(z: &[f64], f: &[f64], zt: f64) -> f64 {
    if zt <= z[0] {
        return f[0];
    }
    for i in 1..z.len() {
        if z[i] >= zt {
            let (z0, z1) = (z[i - 1], z[i]);
            if z1 == z0 {
                return f[i];
            }
            return f[i - 1] + (zt - z0) / (z1 - z0) * (f[i] - f[i - 1]);
        }
    }
    *f.last().unwrap()
}

/// The height-colored trace (SHARPpy-Reimagined `draw_hodo` override: band
/// edges at 0/0.5/3/6/9/12 km AGL). Returns false when there is no wind data.
fn draw_hodo_trace(p: &Painter, g: &Geom, prof: &Profile) -> bool {
    let inner = &prof.inner;
    let n = inner.u.len().min(inner.v.len()).min(inner.hght.len());
    let mut z: Vec<f64> = Vec::new();
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for i in 0..n {
        if inner.u[i].is_finite() && inner.v[i].is_finite() && inner.hght[i].is_finite() {
            let (x, y) = g.uv_to_pix(inner.u[i], inner.v[i]);
            z.push(inner.to_agl(inner.hght[i]));
            xs.push(x);
            ys.push(y);
        }
    }
    if z.len() < 2 {
        return false;
    }
    let zmin = z.iter().cloned().fold(f64::INFINITY, f64::min);
    let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let seg_bnds: Vec<f64> = [0.0, 500.0, 3000.0, 6000.0, 9000.0, 12000.0]
        .iter()
        .map(|b: &f64| b.max(zmin))
        .collect();
    let nseg = seg_bnds.iter().filter(|b| **b <= zmax).count();
    let seg_pos: Vec<(f64, f64)> = seg_bnds[..nseg]
        .iter()
        .map(|b| (interp_z(&z, &xs, *b), interp_z(&z, &ys, *b)))
        .collect();
    // np.searchsorted (left): first index with z[i] >= bnd.
    let seg_idx: Vec<usize> = seg_bnds
        .iter()
        .map(|b| z.partition_point(|v| v < b))
        .collect();

    for idx in 0..nseg.saturating_sub(1) {
        let mut pts = vec![g.pt(seg_pos[idx].0, seg_pos[idx].1)];
        for zi in seg_idx[idx]..seg_idx[idx + 1] {
            pts.push(g.pt(xs[zi], ys[zi]));
        }
        pts.push(g.pt(seg_pos[idx + 1].0, seg_pos[idx + 1].1));
        p.add(Shape::line(pts, Stroke::new(2.0, TRACE_COLORS[idx])));
    }
    // Data top below 12 km AGL: finish the last band with the raw tail.
    if zmax < 12000.0 {
        let idx = nseg - 1;
        let mut pts = vec![g.pt(seg_pos[idx].0, seg_pos[idx].1)];
        for zi in seg_idx[idx]..z.len() {
            pts.push(g.pt(xs[zi], ys[zi]));
        }
        if pts.len() >= 2 {
            p.add(Shape::line(pts, Stroke::new(2.0, TRACE_COLORS[idx])));
        }
    }
    true
}

/// Port of `plotHodo.drawSMV`: Bunkers +'s and circles, effective-inflow
/// lines to the right mover, and the "259/48 RM" / "237/68 LM" labels.
fn draw_smv(p: &Painter, g: &Geom, fonts: &Fonts, prof: &Profile, style: &SkewTStyle) {
    let (rstu, rstv, lstu, lstv) = prof.srwind;
    if !qc(rstu) || !qc(lstu) {
        return;
    }
    let stroke = Stroke::new(1.0, style.fg_color);
    // +'s at the Bunkers vector locations (prof.bunkers == prof.srwind here).
    let (rx, ry) = g.uv_to_pix(rstu, rstv);
    let (lx, ly) = g.uv_to_pix(lstu, lstv);
    for (x, y) in [(rx, ry), (lx, ly)] {
        p.line_segment([g.pt(x - 2.0, y), g.pt(x + 2.0, y)], stroke);
        p.line_segment([g.pt(x, y - 2.0), g.pt(x, y + 2.0)], stroke);
    }
    // Circles around the storm motion vectors.
    p.circle_stroke(g.pt(rx, ry), 5.0, stroke);
    p.circle_stroke(g.pt(lx, ly), 5.0, stroke);

    // Effective inflow layer markers: lines from the right mover to the
    // interpolated wind at the layer bottom/top.
    if qc(prof.etop) && qc(prof.ebottom) {
        let (utop, vtop) = prof.inner.interp_wind(prof.etop);
        let (ubot, vbot) = prof.inner.interp_wind(prof.ebottom);
        let eff = Stroke::new(1.0, EFF_INFLOW_COLOR);
        if qc(ubot) && qc(vbot) {
            let (bx, by) = g.uv_to_pix(ubot, vbot);
            p.line_segment([g.pt(rx, ry), g.pt(bx, by)], eff);
        }
        if qc(utop) && qc(vtop) {
            let (tx, ty) = g.uv_to_pix(utop, vtop);
            p.line_segment([g.pt(rx, ry), g.pt(tx, ty)], eff);
        }
    }

    // Labels in 55x12 rects offset (+2, +5) from the markers.
    let (rm_dir, rm_spd) = comp2vec(rstu, rstv);
    let (lm_dir, lm_spd) = comp2vec(lstu, lstv);
    p.text(
        g.pt(rx + 2.0 + 27.5, ry + 5.0 + 6.0),
        Align2::CENTER_CENTER,
        format!("{}/{} RM", int2str(rm_dir), int2str(rm_spd)),
        fonts.label.clone(),
        style.fg_color,
    );
    p.text(
        g.pt(lx + 2.0 + 27.5, ly + 5.0 + 6.0),
        Align2::CENTER_CENTER,
        format!("{}/{} LM", int2str(lm_dir), int2str(lm_spd)),
        fonts.label.clone(),
        style.fg_color,
    );
}

/// Port of `plotHodo.drawCorfidi`: upshear/downshear circles + labels.
fn draw_corfidi(p: &Painter, g: &Geom, fonts: &Fonts, dv: &DerivedParams) {
    let (up_u, up_v) = dv.corfidi_up;
    let (dn_u, dn_v) = dv.corfidi_dn;
    if !qc(up_u) || !qc(up_v) || !qc(dn_u) || !qc(dn_v) {
        return;
    }
    let stroke = Stroke::new(1.0, CORFIDI_COLOR);
    let (ux, uy) = g.uv_to_pix(up_u, up_v);
    let (dx, dy) = g.uv_to_pix(dn_u, dn_v);
    p.circle_stroke(g.pt(ux, uy), 3.0, stroke);
    p.circle_stroke(g.pt(dx, dy), 3.0, stroke);

    // Labels in 60x10 rects offset (+1, +3).
    let (up_dir, up_spd) = comp2vec(up_u, up_v);
    let (dn_dir, dn_spd) = comp2vec(dn_u, dn_v);
    p.text(
        g.pt(ux + 1.0 + 30.0, uy + 3.0 + 5.0),
        Align2::CENTER_CENTER,
        format!("UP={}/{}", int2str(up_dir), int2str(up_spd)),
        fonts.label.clone(),
        CORFIDI_COLOR,
    );
    p.text(
        g.pt(dx + 1.0 + 30.0, dy + 3.0 + 5.0),
        Align2::CENTER_CENTER,
        format!("DN={}/{}", int2str(dn_dir), int2str(dn_spd)),
        fonts.label.clone(),
        CORFIDI_COLOR,
    );
}

/// Port of `plotHodo.drawLCLtoEL_MW`: the LCL-EL mean wind square + label.
fn draw_lcl_to_el_mw(p: &Painter, g: &Geom, fonts: &Fonts, dv: &DerivedParams) {
    let (mw_dir, mw_spd) = dv.mean_lcl_el;
    if !qc(mw_dir) || !qc(mw_spd) {
        return;
    }
    let (mu, mv) = vec2comp(mw_dir, mw_spd);
    let (mx, my) = g.uv_to_pix(mu, mv);
    // Qt: drawRect(mean_u - 4, mean_v + 4, 8, 8) with a 2 px pen.
    p.rect_stroke(
        Rect::from_min_size(g.pt(mx - 4.0, my + 4.0), Vec2::new(8.0, 8.0)),
        0.0,
        Stroke::new(2.0, MEAN_LCL_EL_COLOR),
        StrokeKind::Middle,
    );
    // Label in a 40x12 rect offset (+1, +5).
    p.text(
        g.pt(mx + 1.0 + 20.0, my + 5.0 + 6.0),
        Align2::CENTER_CENTER,
        format!("{}/{}", int2str(mw_dir), int2str(mw_spd)),
        fonts.label.clone(),
        MEAN_LCL_EL_COLOR,
    );
}

/// Port of `plotHodo.drawCriticalAngle`: the sfc -> 500 m AGL line plus the
/// "Critical Angle = 56 deg" readout at the bottom-left of the frame.
fn draw_critical_angle(
    p: &Painter,
    g: &Geom,
    fonts: &Fonts,
    prof: &Profile,
    dv: &DerivedParams,
    style: &SkewTStyle,
) {
    let inner = &prof.inner;
    let sfc_pres = inner.pres[inner.sfc];
    let pres_500m = inner.pres_at_height(inner.to_msl(500.0));
    let (sfc_u, sfc_v) = inner.interp_wind(sfc_pres);
    let (u500, v500) = if pres_500m.is_finite() {
        inner.interp_wind(pres_500m)
    } else {
        (f64::NAN, f64::NAN)
    };
    // Only drawn when the effective inflow layer is surface based.
    if !qc(prof.etop)
        || !qc(prof.ebottom)
        || prof.ebottom != sfc_pres
        || !qc(sfc_u)
        || !qc(sfc_v)
        || !qc(u500)
        || !qc(v500)
    {
        return;
    }
    let (sx, sy) = g.uv_to_pix(sfc_u, sfc_v);
    let (x5, y5) = g.uv_to_pix(u500, v500);
    p.line_segment(
        [g.pt(sx, sy), g.pt(x5, y5)],
        Stroke::new(1.0, CRIT_LINE_COLOR),
    );

    let (rstu, _, lstu, _) = prof.srwind;
    if qc(rstu) && qc(lstu) && qc(dv.right_critical_angle) {
        // Half-alpha bg blank behind the text (setBlackPen), then the readout.
        let bg = style.bg_color;
        p.rect_filled(
            Rect::from_min_size(
                g.pt(15.0, g.hgt - 36.0),
                Vec2::new(100.0, (fonts.critical_height + 5.0) as f32),
            ),
            0.0,
            Color32::from_rgba_unmultiplied(bg.r(), bg.g(), bg.b(), 128),
        );
        p.text(
            g.pt(15.0, g.hgt - 36.0),
            Align2::LEFT_TOP,
            format!(
                "Critical Angle = {}\u{00B0}",
                int2str(dv.right_critical_angle)
            ),
            fonts.critical.clone(),
            CRIT_TEXT_COLOR,
        );
    }
}

// ----------------------------------------------------------------------
// Locator inset (port of sharpmod.viz.hodo_locator.draw_hodo_locator)
// ----------------------------------------------------------------------

/// Geographic extent of the inset. The Python original shows a ~1.4 deg-tall
/// county view fetched from TIGERweb; with the embedded state-boundary
/// basemap the window is widened so state lines are actually in view.
fn zoom_bounds(lat: f64, lon: f64) -> (f64, f64, f64, f64) {
    let half_lat = 4.0;
    let cos_lat = lat.to_radians().cos().max(0.35);
    let half_lon = half_lat * 1.35 / cos_lat;
    let ext = basemap::widen_until_outline(
        basemap::Extent {
            west: lon - half_lon,
            south: lat - half_lat,
            east: lon + half_lon,
            north: lat + half_lat,
        },
        lon,
        lat,
    );
    (ext.west, ext.south, ext.east, ext.north)
}

fn draw_locator(p: &Painter, g: &Geom, prof: &Profile, style: &SkewTStyle) {
    let lat = prof.inner.station.latitude;
    let lon = prof.inner.station.longitude;
    if !lat.is_finite() || !lon.is_finite() || lat.abs() > 90.0 || lon.abs() > 180.0 {
        return;
    }
    // Share the hodograph's upper-left corner (frame border + 1 px).
    let (frame_left, frame_top) = (1.0f64, 1.0f64);
    let avail_w = (g.wid - frame_left - 8.0).max(0.0);
    let avail_h = (g.hgt - frame_top - 8.0).max(0.0);
    let width = (g.wid * 0.29).floor().max(150.0).min(250.0).min(avail_w);
    let height = (width * 0.64).floor().max(96.0).min(avail_h);
    if width < 110.0 || height < 72.0 {
        return;
    }
    let rect = Rect::from_min_size(
        g.pt(frame_left, frame_top),
        Vec2::new(width as f32, height as f32),
    );
    // The map outlines use the selected foreground; match its background.
    p.rect_filled(
        rect,
        0.0,
        if style.bg_color == Color32::BLACK {
            MAP_FILL
        } else {
            style.bg_color
        },
    );
    p.rect_stroke(
        rect,
        0.0,
        Stroke::new(1.25, style.fg_color),
        StrokeKind::Middle,
    );

    let interior = rect.shrink(5.0);
    let mp = p.with_clip_rect(interior);
    let (west, south, east, north) = zoom_bounds(lat, lon);
    let map_point = |plon: f64, plat: f64| -> Pos2 {
        Pos2::new(
            interior.min.x + ((plon - west) / (east - west) * interior.width() as f64) as f32,
            interior.min.y + ((north - plat) / (north - south) * interior.height() as f64) as f32,
        )
    };

    // Coastline / border / state polylines (embedded world basemap).
    basemap::draw(
        &mp,
        basemap::Extent {
            west,
            south,
            east,
            north,
        },
        map_point,
        Stroke::new(1.0, style.fg_color),
    );

    // Crosshair at the sounding point.
    let c = map_point(lon, lat);
    let marker = Stroke::new(1.4, MAP_POINT_COLOR);
    mp.circle(c, 4.0, MAP_FILL, marker);
    mp.line_segment([c - Vec2::new(7.0, 0.0), c + Vec2::new(7.0, 0.0)], marker);
    mp.line_segment([c - Vec2::new(0.0, 7.0), c + Vec2::new(0.0, 7.0)], marker);
}

/// Linked cursor marker: the wind at `h_agl` (m) highlighted on the hodograph
/// with a wind/height readout, driven by hovering the skew-T (see
/// `SoundingView`).
pub(crate) fn cursor_marker(
    painter: &Painter,
    rect: Rect,
    prof: &Profile,
    style: &SkewTStyle,
    h_agl: f64,
    zoom_kts: f64,
) {
    let g = Geom::with_zoom(rect, zoom_kts);
    let inner = &prof.inner;
    let pres = inner.pres_at_height(inner.to_msl(h_agl));
    if !pres.is_finite() {
        return;
    }
    let (u, v) = inner.interp_wind(pres.min(inner.pres[inner.sfc]));
    if !u.is_finite() || !v.is_finite() {
        return;
    }
    let p = painter.with_clip_rect(rect);
    let (x, y) = g.uv_to_pix(u, v);
    let c = g.pt(x, y);
    p.circle_stroke(c, 5.0, Stroke::new(1.6, style.fg_color));
    p.circle_stroke(c, 1.0, Stroke::new(2.0, style.fg_color));

    let (wdir, wspd) = inner.interp_vec(pres.min(inner.pres[inner.sfc]));
    let text = format!(
        "{:.1} km  {}/{}",
        h_agl / 1000.0,
        crate::utils::int2str(wdir),
        crate::utils::int2str(wspd)
    );
    let font = style.regular_font(11.0);
    let galley = p.layout_no_wrap(text, font, style.fg_color);
    let mut anchor = c + Vec2::new(8.0, -8.0 - galley.size().y);
    if anchor.x + galley.size().x > rect.max.x - 2.0 {
        anchor.x = c.x - 8.0 - galley.size().x;
    }
    if anchor.y < rect.min.y + 2.0 {
        anchor.y = c.y + 8.0;
    }
    let bg = Rect::from_min_size(anchor, galley.size() + Vec2::new(4.0, 2.0));
    p.rect_filled(bg, 0.0, style.bg_color);
    p.galley(bg.min + Vec2::new(2.0, 1.0), galley, style.fg_color);
}
