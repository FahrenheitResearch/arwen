//! The SPC-style Skew-T widget: a faithful egui port of
//! `sharppy.viz.skew.plotSkewT` (SHARPpy 1.4.0a5) plus the
//! SHARPpy-Reimagined overlay passes (CAPE/CIN buoyancy fill, HGZ band,
//! speed-colored wind barbs, Space Grotesk styling).

use egui::{
    Align2, Color32, FontFamily, FontId, Painter, Pos2, Rect, Response, Sense, Shape, Stroke, Ui,
    Vec2, Widget,
};

use sharprs::params::cape::ParcelResult;
use sharprs::thermo;

use crate::barbs::draw_barb;
use crate::profile::{ParcelType, Profile};
use crate::utils::{int2str, m2ft, qc};

/// Colors of the skew-T (defaults match the SPC dark scheme of the original).
#[derive(Clone, Debug)]
pub struct SkewTStyle {
    pub bg_color: Color32,
    pub fg_color: Color32,
    pub isotherm_color: Color32,
    pub isotherm_hgz_color: Color32,
    pub adiab_color: Color32,
    pub mixr_color: Color32,
    pub temp_color: Color32,
    pub dewp_color: Color32,
    pub wetbulb_color: Color32,
    pub eff_layer_color: Color32,
    pub hgt_color: Color32,
    pub lcl_mkr_color: Color32,
    pub lfc_mkr_color: Color32,
    pub el_mkr_color: Color32,
    pub sig_temp_level_color: Color32,
    pub cape_fill_color: Color32,
    pub cin_fill_color: Color32,
    pub hgz_edge_color: Color32,
    pub omega_up_color: Color32,
    pub omega_down_color: Color32,
    pub omega_frame_color: Color32,
    pub downdraft_color: Color32,
    /// Alert colors for the max-lapse-rate readout (dbrown, lbrown, white,
    /// yellow, red, purple).
    pub alert_colors: [Color32; 6],
    /// Regular / bold font families. `install_fonts` registers Space Grotesk;
    /// [`SkewTStyle::space_grotesk`] selects it.
    pub font_regular: FontFamily,
    pub font_bold: FontFamily,
    /// Independent multiplier for every sounding `FontId`. Panel and line
    /// geometry remain in their original coordinate system.
    pub text_scale: f32,
}

/// Cross-platform font pairs that never depend on operating-system fonts.
/// Space Grotesk is bundled by this crate; the other choices use egui's
/// bundled fallback faces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoundingFontPreset {
    SpaceGrotesk,
    CleanProportional,
    TechnicalMonospace,
}

impl Default for SkewTStyle {
    fn default() -> Self {
        SkewTStyle {
            bg_color: Color32::BLACK,
            fg_color: Color32::WHITE,
            isotherm_color: Color32::from_rgb(0x55, 0x55, 0x55),
            isotherm_hgz_color: Color32::from_rgb(0x00, 0x00, 0xFF),
            adiab_color: Color32::from_rgb(0x33, 0x33, 0x33),
            mixr_color: Color32::from_rgb(0x00, 0x66, 0x00),
            temp_color: Color32::from_rgb(0xFF, 0x00, 0x00),
            dewp_color: Color32::from_rgb(0x00, 0xFF, 0x00),
            wetbulb_color: Color32::from_rgb(0x00, 0xFF, 0xFF),
            eff_layer_color: Color32::from_rgb(0x04, 0xDB, 0xD8),
            hgt_color: Color32::from_rgb(0xFF, 0x00, 0x00),
            lcl_mkr_color: Color32::from_rgb(0x00, 0xFF, 0x00),
            lfc_mkr_color: Color32::from_rgb(0xFF, 0xFF, 0x00),
            el_mkr_color: Color32::from_rgb(0xFF, 0x00, 0xFF),
            sig_temp_level_color: Color32::from_rgb(0x0A, 0x63, 0xFF),
            cape_fill_color: Color32::from_rgba_unmultiplied(255, 130, 0, 80),
            cin_fill_color: Color32::from_rgba_unmultiplied(30, 120, 255, 70),
            hgz_edge_color: Color32::from_rgba_unmultiplied(0, 200, 255, 100),
            omega_up_color: Color32::from_rgb(0xFF, 0x66, 0x66),
            omega_down_color: Color32::from_rgb(0x00, 0x66, 0xCC),
            omega_frame_color: Color32::from_rgb(0xFF, 0x00, 0xFF),
            downdraft_color: Color32::from_rgb(0xFF, 0x00, 0xFF),
            alert_colors: [
                Color32::from_rgb(0x77, 0x50, 0x00),
                Color32::from_rgb(0x99, 0x66, 0x00),
                Color32::from_rgb(0xFF, 0xFF, 0xFF),
                Color32::from_rgb(0xFF, 0xFF, 0x00),
                Color32::from_rgb(0xFF, 0x00, 0x00),
                Color32::from_rgb(0xE7, 0x00, 0xDF),
            ],
            font_regular: FontFamily::Proportional,
            font_bold: FontFamily::Proportional,
            text_scale: 1.0,
        }
    }
}

impl SkewTStyle {
    /// Style using the bundled Space Grotesk families (call
    /// [`crate::install_fonts`] once first).
    pub fn space_grotesk() -> Self {
        SkewTStyle {
            font_regular: FontFamily::Name(crate::FONT_FAMILY.into()),
            font_bold: FontFamily::Name(crate::FONT_FAMILY_BOLD.into()),
            ..Default::default()
        }
    }

    /// Select one of the fully bundled/cross-platform sounding font pairs.
    pub fn with_font_preset(mut self, preset: SoundingFontPreset) -> Self {
        (self.font_regular, self.font_bold) = match preset {
            SoundingFontPreset::SpaceGrotesk => (
                FontFamily::Name(crate::FONT_FAMILY.into()),
                FontFamily::Name(crate::FONT_FAMILY_BOLD.into()),
            ),
            SoundingFontPreset::CleanProportional => {
                (FontFamily::Proportional, FontFamily::Proportional)
            }
            SoundingFontPreset::TechnicalMonospace => {
                (FontFamily::Monospace, FontFamily::Monospace)
            }
        };
        self
    }

    /// Set the independent sounding text scale. Invalid/extreme input is
    /// normalized so persisted settings cannot hide text or request
    /// pathological glyph sizes.
    pub fn with_text_scale(mut self, scale: f32) -> Self {
        self.text_scale = normalized_text_scale(scale);
        self
    }

    /// Construct a regular font using this style's family and text scale.
    pub fn regular_font(&self, base_size: f32) -> FontId {
        FontId::new(
            base_size * normalized_text_scale(self.text_scale),
            self.font_regular.clone(),
        )
    }

    /// Construct a bold font using this style's family and text scale.
    pub fn bold_font(&self, base_size: f32) -> FontId {
        FontId::new(
            base_size * normalized_text_scale(self.text_scale),
            self.font_bold.clone(),
        )
    }
}

fn normalized_text_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.clamp(0.5, 2.0)
    } else {
        1.0
    }
}

/// The Skew-T widget. Create with [`SkewT::new`], configure with the builder
/// methods, then `ui.add(skewt)`.
pub struct SkewT<'a> {
    prof: &'a Profile,
    parcel: ParcelType,
    title: String,
    brand: Option<String>,
    plot_omega: Option<bool>,
    interp_winds: bool,
    style: SkewTStyle,
    size: Option<Vec2>,
    cursor_readout: bool,
}

impl<'a> SkewT<'a> {
    pub fn new(prof: &'a Profile) -> Self {
        SkewT {
            prof,
            parcel: ParcelType::MostUnstable,
            title: String::new(),
            brand: None,
            plot_omega: None,
            interp_winds: true,
            style: SkewTStyle::default(),
            size: None,
            cursor_readout: true,
        }
    }

    /// Enable/disable the hover readout cursor (default: on).
    pub fn cursor_readout(mut self, on: bool) -> Self {
        self.cursor_readout = on;
        self
    }

    /// Title drawn at the top-left (e.g. `"Forecast profile"`).
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Optional small brand text drawn at the top-right.
    pub fn brand(mut self, brand: impl Into<String>) -> Self {
        self.brand = Some(brand.into());
        self
    }

    /// Which parcel to display (default: most unstable, like the original
    /// renderer).
    pub fn parcel(mut self, parcel: ParcelType) -> Self {
        self.parcel = parcel;
        self
    }

    /// Force the omega meter on/off (default: on when omega data exists).
    pub fn plot_omega(mut self, on: bool) -> Self {
        self.plot_omega = Some(on);
        self
    }

    /// Draw barbs at raw levels instead of every 40 hPa (default: interpolated).
    pub fn interp_winds(mut self, on: bool) -> Self {
        self.interp_winds = on;
        self
    }

    pub fn style(mut self, style: SkewTStyle) -> Self {
        self.style = style;
        self
    }

    /// Fixed size; defaults to all available space.
    pub fn size(mut self, size: Vec2) -> Self {
        self.size = Some(size);
        self
    }
}

/// Geometry + transforms (port of `backgroundSkewT.initUI`).
struct Geom {
    rect: Rect,
    lpad: f64,
    rpad: f64,
    tpad: f64,
    bpad: f64,
    wid: f64,
    hgt: f64,
    brx: f64,
    bry: f64,
    barbx: f64,
    pmax: f64,
    pmin: f64,
    log_pmax: f64,
    log_pmin: f64,
    bltmpc: f64,
    brtmpc: f64,
    dt: f64,
    xrange: f64,
    yrange: f64,
}

impl Geom {
    fn new(rect: Rect) -> Geom {
        let w = rect.width() as f64;
        let h = rect.height() as f64;
        // SKEWT_LPAD=46 from the SHARPpy-Reimagined renderer; the rest are the
        // vendored defaults.
        let (lpad, rpad, tpad, bpad) = (46.0, 65.0, 20.0, 20.0);
        let wid = w - rpad;
        let hgt = h - bpad;
        let xskew = 100.0f64 / 3.0;
        let xrange = 100.0;
        Geom {
            rect,
            lpad,
            rpad,
            tpad,
            bpad,
            wid,
            hgt,
            brx: wid,
            bry: hgt,
            barbx: wid + rpad / 2.0,
            pmax: 1050.0,
            pmin: 100.0,
            log_pmax: 1050.0f64.ln(),
            log_pmin: 100.0f64.ln(),
            bltmpc: -50.0,
            brtmpc: 50.0,
            dt: 10.0,
            xrange,
            yrange: xskew.to_radians().tan() * xrange,
        }
    }

    /// Pressure (hPa) -> y (widget-local px).
    fn pres_to_pix(&self, p: f64) -> f64 {
        let scl1 = self.log_pmax - self.log_pmin;
        let scl2 = self.log_pmax - p.ln();
        self.bry - (scl2 / scl1) * (self.bry - self.tpad)
    }

    /// y (widget-local px) -> pressure (hPa); exact inverse of `pres_to_pix`.
    fn pix_to_pres(&self, y: f64) -> f64 {
        let scl1 = self.log_pmax - self.log_pmin;
        let frac = (self.bry - y) / (self.bry - self.tpad);
        (self.log_pmax - frac * scl1).exp()
    }

    /// (temperature C, pressure hPa) -> x (widget-local px).
    fn tmpc_to_pix(&self, t: f64, p: f64) -> f64 {
        let scl1 =
            self.brtmpc - ((self.bry - self.pres_to_pix(p)) / (self.bry - self.tpad)) * self.yrange;
        self.brx - ((scl1 - t) / self.xrange) * (self.brx - self.lpad)
    }

    /// Widget-local -> screen position.
    fn pt(&self, x: f64, y: f64) -> Pos2 {
        Pos2::new(self.rect.min.x + x as f32, self.rect.min.y + y as f32)
    }

    /// Screen-space rect from widget-local coordinates.
    fn local_rect(&self, x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect::from_min_size(self.pt(x, y), Vec2::new(w as f32, h as f32))
    }
}

/// Fonts sized from the widget height like the Qt original (pt -> px at the
/// standard 96-dpi factor 4/3).
struct Fonts {
    title: FontId,
    label: FontId,
    label_bold: FontId,
    env_trace: FontId,
    in_plot: FontId,
    esrh: FontId,
    hght: FontId,
    hgz: FontId,
    esrh_height: f64,
    title_height: f64,
}

impl Fonts {
    fn new(hgt: f64, style: &SkewTStyle) -> Fonts {
        const PT: f64 = 4.0 / 3.0;
        let fsize = 6.0;
        let fsizet = 10.0;
        // TITLE_FONT_SCALE = 0.90 from the SHARPpy-Reimagined renderer.
        let title_pt = (fsizet + hgt * 0.006) * 0.90;
        let label_pt = fsize + 2.0 + hgt * 0.0045;
        let esrh_pt = fsize + 2.0 + hgt * 0.0045;
        Fonts {
            title: style.regular_font((title_pt * PT) as f32),
            label: style.regular_font((label_pt * PT) as f32),
            label_bold: style.bold_font((label_pt * PT) as f32),
            env_trace: style.regular_font(((11.0 + hgt * 0.0045) * PT) as f32),
            in_plot: style.regular_font(((fsize + hgt * 0.0045) * PT) as f32),
            esrh: style.regular_font((esrh_pt * PT) as f32),
            hght: style.regular_font(((9.0 + hgt * 0.0045) * PT) as f32),
            hgz: style.bold_font((9.0 * PT) as f32),
            esrh_height: esrh_pt * PT * 0.5 + 9.0 + hgt * 0.0045,
            title_height: title_pt * PT * 0.5 + 5.0 + hgt * 0.003,
        }
    }
}

impl Widget for SkewT<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let size = self.size.unwrap_or_else(|| ui.available_size());
        let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
        if !ui.is_rect_visible(rect) || rect.width() < 50.0 || rect.height() < 50.0 {
            return response;
        }
        let g = Geom::new(rect);
        let fonts = Fonts::new(g.hgt, &self.style);
        let painter = ui.painter_at(rect);

        self.draw_background(&painter, &g, &fonts);
        self.draw_data(&painter, &g, &fonts);
        if self.cursor_readout
            && let Some(pos) = response.hover_pos()
            && let Some(pres) = hover_pressure(rect, pos)
        {
            self.draw_readout(&painter, &g, pres);
        }
        response
    }
}

impl SkewT<'_> {
    fn plot_omega_on(&self) -> bool {
        self.plot_omega
            .unwrap_or_else(|| self.prof.inner.omeg.iter().any(|o| o.is_finite()))
    }

    // ------------------------------------------------------------------
    // Background pass (port of backgroundSkewT.plotBackground)
    // ------------------------------------------------------------------
    fn draw_background(&self, painter: &Painter, g: &Geom, fonts: &Fonts) {
        let st = &self.style;
        painter.rect_filled(g.rect, 0.0, st.bg_color);

        // Grid clipped to the plot area (the original lets it spill under the
        // barb strip and then blanks that margin; same net result).
        let plot_clip = Rect::from_min_max(g.pt(g.lpad, g.tpad), g.pt(g.brx, g.bry));
        let gp = painter.with_clip_rect(plot_clip);

        // Isotherms.
        let mut t = g.bltmpc - 100.0;
        while t <= g.brtmpc + 1e-9 {
            let x1 = g.tmpc_to_pix(t, g.pmax);
            let x2 = g.tmpc_to_pix(t, g.pmin);
            let color = if t == 0.0 || t == -20.0 {
                st.isotherm_hgz_color
            } else {
                st.isotherm_color
            };
            gp.extend(Shape::dashed_line(
                &[g.pt(x1, g.bry), g.pt(x2, g.tpad)],
                Stroke::new(1.0, color),
                4.0,
                2.0,
            ));
            t += g.dt;
        }

        // Dry adiabats.
        let mut theta = g.bltmpc;
        while theta < 80.0 {
            let mut pts = Vec::new();
            let mut p = g.pmax;
            while p >= g.pmin {
                let t = (theta + 273.15) / (1000.0 / p).powf(sharprs::constants::ROCP) - 273.15;
                pts.push(g.pt(g.tmpc_to_pix(t, p), g.pres_to_pix(p)));
                p -= 10.0;
            }
            gp.add(Shape::line(pts, Stroke::new(1.0, st.adiab_color)));
            theta += 20.0;
        }

        // Mixing ratio lines + labels. The original's `[2] + np.arange(4, 33, 4)`
        // is a numpy broadcast (not list concat), yielding this set.
        let ws = [6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0];
        for w in ws {
            let t1 = thermo::temp_at_mixrat(w, g.pmax);
            let x1 = g.tmpc_to_pix(t1, g.pmax);
            let y1 = g.pres_to_pix(g.pmax);
            let t2 = thermo::temp_at_mixrat(w, 600.0);
            let x2 = g.tmpc_to_pix(t2, 600.0);
            let y2 = g.pres_to_pix(600.0);
            let text = int2str(w);
            let galley = painter.layout_no_wrap(text.clone(), fonts.in_plot.clone(), st.mixr_color);
            let (tw, th) = (galley.size().x as f64 + 4.0, galley.size().y as f64);
            gp.rect_filled(
                g.local_rect(x2 - tw / 2.0, y2 - th, tw, th),
                0.0,
                st.bg_color,
            );
            gp.line_segment(
                [g.pt(x1, y1), g.pt(x2, y2)],
                Stroke::new(1.0, st.mixr_color),
            );
            gp.text(
                g.pt(x2, y2),
                Align2::CENTER_BOTTOM,
                text,
                fonts.in_plot.clone(),
                st.mixr_color,
            );
        }

        // Frame border.
        let border = Stroke::new(2.0, st.fg_color);
        let w = g.wid + g.rpad;
        painter.line_segment([g.pt(g.lpad, g.tpad), g.pt(w, g.tpad)], border);
        painter.line_segment([g.pt(w, g.tpad), g.pt(w, g.bry)], border);
        painter.line_segment([g.pt(w, g.bry), g.pt(g.lpad, g.bry)], border);
        painter.line_segment([g.pt(g.lpad, g.bry), g.pt(g.lpad, g.tpad)], border);

        // Major isobars + labels.
        for p in [1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0] {
            let y = g.pres_to_pix(p);
            if y >= g.tpad && y <= g.hgt {
                painter.line_segment(
                    [g.pt(g.lpad, y), g.pt(g.brx, y)],
                    Stroke::new(1.0, st.fg_color),
                );
                painter.text(
                    g.pt(g.lpad - 3.0, y),
                    Align2::RIGHT_CENTER,
                    int2str(p),
                    fonts.label_bold.clone(),
                    st.fg_color,
                );
            }
        }

        // Isotherm labels along the bottom.
        let mut t = g.bltmpc;
        while t <= g.brtmpc + 1e-9 {
            let x = g.tmpc_to_pix(t, g.pmax);
            if x >= g.lpad && x <= g.wid {
                painter.text(
                    g.pt(x, g.bry + 2.0),
                    Align2::CENTER_TOP,
                    int2str(t),
                    fonts.label_bold.clone(),
                    st.fg_color,
                );
            }
            t += g.dt;
        }

        // Minor isobar ticks.
        let mut p = g.pmax;
        while p >= g.pmin - 1e-9 {
            let y = g.pres_to_pix(p);
            if y >= g.tpad && y <= g.hgt {
                painter.line_segment(
                    [g.pt(g.lpad, y), g.pt(g.lpad + 5.0, y)],
                    Stroke::new(1.0, st.fg_color),
                );
                painter.line_segment(
                    [g.pt(w - 5.0, y), g.pt(w, y)],
                    Stroke::new(1.0, st.fg_color),
                );
            }
            p -= 50.0;
        }
    }

    // ------------------------------------------------------------------
    // Data pass (port of plotSkewT.plotData + the overlay wrappers)
    // ------------------------------------------------------------------
    fn draw_data(&self, painter: &Painter, g: &Geom, fonts: &Fonts) {
        let st = &self.style;
        let prof = self.prof;
        let pcl = prof.parcel(self.parcel);

        // The vendored clip: x in [lpad, brx+rpad], y in [tpad, bry].
        let clip = Rect::from_min_max(g.pt(g.lpad, g.tpad), g.pt(g.wid + g.rpad, g.bry));
        let dp = painter.with_clip_rect(clip);

        self.draw_titles(painter, g, fonts);

        // Wetbulb, temperature, virtual temperature traces.
        self.draw_trace(
            &dp,
            painter,
            g,
            fonts,
            &prof.inner.wetbulb,
            &prof.inner.pres,
            st.wetbulb_color,
            1.0,
            false,
            true,
        );
        self.draw_trace(
            &dp,
            painter,
            g,
            fonts,
            &prof.inner.tmpc,
            &prof.inner.pres,
            st.temp_color,
            3.0,
            false,
            true,
        );
        self.draw_trace(
            &dp,
            painter,
            g,
            fonts,
            &prof.inner.vtmp,
            &prof.inner.pres,
            st.temp_color,
            1.0,
            true,
            false,
        );

        // Max lapse rate + significant temperature levels.
        self.draw_max_lapse_rate_layer(&dp, painter, g, fonts);
        self.draw_temp_levels(&dp, g, fonts, pcl);

        // Dewpoint trace.
        self.draw_trace(
            &dp,
            painter,
            g,
            fonts,
            &prof.inner.dwpc,
            &prof.inner.pres,
            st.dewp_color,
            3.0,
            false,
            true,
        );

        // Height markers.
        for h in [0.0, 1000.0, 3000.0, 6000.0, 9000.0, 12000.0, 15000.0] {
            self.draw_height(&dp, g, fonts, h);
        }

        // Parcel + downdraft parcel traces.
        self.draw_virtual_parcel_trace(&dp, g, &pcl.ttrace, &pcl.ptrace, st.fg_color);
        self.draw_virtual_parcel_trace(
            &dp,
            g,
            &prof.dpcl_ttrace,
            &prof.dpcl_ptrace,
            st.downdraft_color,
        );

        // LCL / LFC / EL markers.
        self.draw_parcel_levels(&dp, g, fonts, pcl);

        // Wind barbs (clip extended by bpad at the bottom like the original).
        let barb_clip =
            Rect::from_min_max(g.pt(g.lpad, g.tpad), g.pt(g.wid + g.rpad, g.bry + g.bpad));
        let bp = painter.with_clip_rect(barb_clip);
        self.draw_barbs(&bp, g);

        // Effective inflow layer.
        self.draw_effective_layer(&dp, g, fonts);

        // Omega profile.
        if self.plot_omega_on() {
            self.draw_omega_profile(&dp, g, fonts);
        }

        // CAPE/CIN buoyancy fill (attached wrapper pass; clip starts at rpad).
        let fill_clip = Rect::from_min_max(g.pt(g.rpad, g.tpad), g.pt(g.brx, g.bry));
        let fp = painter.with_clip_rect(fill_clip);
        self.draw_cape_fill(&fp, g, pcl);

        // HGZ overlay (boundaries + label only).
        self.draw_hgz(painter, g, fonts);
    }

    fn draw_titles(&self, painter: &Painter, g: &Geom, fonts: &Fonts) {
        let st = &self.style;
        // The title lives in the band above the plot border (TITLE_TOP = 3 in
        // the original renderer); anchor to the band's bottom so it never
        // collides with the border at small widget sizes.
        // The whole title is laid out, never cut at the panel edge: it shrinks to the width the panel leaves
        // (down to half size), and a title still too long breaks at the " · " nearest its middle onto a second line, both
        // lines shrunk to fit the band above the plot. Origin: "Ensemble mean (36 me" cut off at the skew-T's right edge (2026-10-24).
        let anchor = g.pt(g.lpad, g.tpad - 2.0);
        let avail = (g.rect.max.x - anchor.x - 6.0).max(40.0);
        let sized = |text: &str, scale: f32| {
            let mut f = fonts.title.clone();
            f.size *= scale;
            painter.layout_no_wrap(text.to_string(), f, st.fg_color)
        };
        let full = sized(&self.title, 1.0);
        let fit = (avail / full.size().x.max(1.0)).min(1.0);
        if fit >= 0.5 || !self.title.contains(" · ") {
            let galley = if fit < 1.0 { sized(&self.title, fit.max(0.35)) } else { full };
            let h = galley.size().y;
            painter.galley(egui::pos2(anchor.x, anchor.y - h), galley, st.fg_color);
        } else {
            let parts: Vec<&str> = self.title.split(" · ").collect();
            let total = self.title.chars().count();
            // The break nearest the middle of the text.
            let mut best = 1;
            let mut seen = 0usize;
            let mut best_gap = usize::MAX;
            for (i, part) in parts.iter().enumerate().take(parts.len() - 1) {
                seen += part.chars().count() + 3;
                let gap = seen.abs_diff(total / 2);
                if gap < best_gap {
                    best_gap = gap;
                    best = i + 1;
                }
            }
            let (a, b) = (parts[..best].join(" · "), parts[best..].join(" · "));
            let (la, lb) = (sized(&a, 1.0), sized(&b, 1.0));
            // Both lines inside the band above the plot: as wide as the panel allows, as tall as the band allows.
            let band = (anchor.y - g.rect.min.y).max(8.0);
            let scale = (avail / la.size().x.max(lb.size().x).max(1.0))
                .min(band / (la.size().y + lb.size().y).max(1.0))
                .min(0.7);
            let (ga, gb) = (sized(&a, scale), sized(&b, scale));
            let (ha, hb) = (ga.size().y, gb.size().y);
            painter.galley(egui::pos2(anchor.x, anchor.y - hb - ha), ga, st.fg_color);
            painter.galley(egui::pos2(anchor.x, anchor.y - hb), gb, st.fg_color);
        }
        if let Some(brand) = &self.brand {
            painter.text(
                g.pt(g.wid + g.rpad, g.tpad - 2.0),
                Align2::RIGHT_BOTTOM,
                brand,
                fonts.label.clone(),
                st.fg_color,
            );
        }
        let _ = fonts.title_height;
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_trace(
        &self,
        dp: &Painter,
        painter: &Painter,
        g: &Geom,
        fonts: &Fonts,
        data: &[f64],
        pres: &[f64],
        color: Color32,
        width: f32,
        dashed: bool,
        label: bool,
    ) {
        let st = &self.style;
        let mut pts: Vec<Pos2> = Vec::new();
        let mut first: Option<(f64, f64)> = None;
        for i in 0..data.len().min(pres.len()) {
            if data[i].is_finite() && pres[i].is_finite() {
                let x = g.tmpc_to_pix(data[i], pres[i]);
                let y = g.pres_to_pix(pres[i]);
                if first.is_none() {
                    first = Some((data[i], x));
                }
                pts.push(g.pt(x, y));
            }
        }
        if pts.len() < 2 {
            return;
        }
        let stroke = Stroke::new(width, color);
        if dashed {
            dp.extend(Shape::dashed_line(&pts, stroke, 5.0, 3.0));
        } else {
            dp.add(Shape::line(pts.clone(), stroke));
        }

        if label {
            if let Some((val, x0)) = first {
                let y0 = g.pres_to_pix(
                    pres.iter()
                        .cloned()
                        .find(|p| p.is_finite())
                        .unwrap_or(1000.0),
                );
                // Surface units default: Fahrenheit (like the original config).
                let text = int2str(thermo::ctof(val));
                let galley = painter.layout_no_wrap(text.clone(), fonts.env_trace.clone(), color);
                let (tw, th) = (galley.size().x as f64 + 4.0, galley.size().y as f64 + 2.0);
                // Unclipped, like the original (setClipping(False)).
                painter.rect_filled(
                    g.local_rect(x0 - tw / 2.0, y0 + 4.0, tw, th),
                    0.0,
                    st.bg_color,
                );
                painter.text(
                    g.pt(x0, y0 + 4.0 + th / 2.0),
                    Align2::CENTER_CENTER,
                    text,
                    fonts.env_trace.clone(),
                    color,
                );
            }
        }
    }

    fn draw_virtual_parcel_trace(
        &self,
        dp: &Painter,
        g: &Geom,
        ttrace: &[f64],
        ptrace: &[f64],
        color: Color32,
    ) {
        if ttrace.is_empty() || ptrace.is_empty() {
            return;
        }
        let mut pts = Vec::new();
        for i in 0..ttrace.len().min(ptrace.len()) {
            if ttrace[i].is_finite() && ptrace[i].is_finite() {
                pts.push(g.pt(
                    g.tmpc_to_pix(ttrace[i], ptrace[i]),
                    g.pres_to_pix(ptrace[i]),
                ));
            }
        }
        if pts.len() >= 2 {
            dp.extend(Shape::dashed_line(&pts, Stroke::new(1.0, color), 5.0, 3.0));
        }
    }

    fn draw_height(&self, dp: &Painter, g: &Geom, fonts: &Fonts, h: f64) {
        let st = &self.style;
        let prof = self.prof;
        let sfc = prof.inner.interp_hght(prof.inner.pres[prof.inner.sfc]);
        let p1 = prof.inner.pres_at_height(h + sfc);
        if !p1.is_finite() {
            return;
        }
        let y1 = g.pres_to_pix(p1);
        dp.line_segment(
            [g.pt(g.lpad, y1), g.pt(g.lpad + 10.0, y1)],
            Stroke::new(1.0, st.hgt_color),
        );
        dp.text(
            g.pt(g.lpad + 15.0, y1),
            Align2::LEFT_CENTER,
            format!("{} km", int2str(h / 1000.0)),
            fonts.hght.clone(),
            st.hgt_color,
        );
    }

    fn draw_parcel_levels(&self, dp: &Painter, g: &Geom, fonts: &Fonts, pcl: &ParcelResult) {
        let st = &self.style;
        let x0 = g.tmpc_to_pix(37.0, 1000.0);
        let x1 = g.tmpc_to_pix(41.0, 1000.0);
        let xc = (x0 + x1) / 2.0;
        // LCL
        if qc(pcl.lclpres) {
            let y = g.pres_to_pix(pcl.lclpres);
            dp.line_segment(
                [g.pt(x0, y), g.pt(x1, y)],
                Stroke::new(2.0, st.lcl_mkr_color),
            );
            dp.text(
                g.pt(xc, y + 8.0),
                Align2::CENTER_CENTER,
                "LCL",
                fonts.hght.clone(),
                st.lcl_mkr_color,
            );
        }
        // LFC
        if qc(pcl.lfcpres) {
            let y = g.pres_to_pix(pcl.lfcpres);
            dp.line_segment(
                [g.pt(x0, y), g.pt(x1, y)],
                Stroke::new(2.0, st.lfc_mkr_color),
            );
            dp.text(
                g.pt(xc, y - 6.0),
                Align2::CENTER_CENTER,
                "LFC",
                fonts.hght.clone(),
                st.lfc_mkr_color,
            );
        }
        // EL
        if qc(pcl.elpres) && pcl.elpres != pcl.lclpres {
            let y = g.pres_to_pix(pcl.elpres);
            dp.line_segment(
                [g.pt(x0, y), g.pt(x1, y)],
                Stroke::new(2.0, st.el_mkr_color),
            );
            dp.text(
                g.pt(xc, y - 6.0),
                Align2::CENTER_CENTER,
                "EL",
                fonts.hght.clone(),
                st.el_mkr_color,
            );
        }
    }

    fn draw_temp_levels(&self, dp: &Painter, g: &Geom, fonts: &Fonts, pcl: &ParcelResult) {
        let st = &self.style;
        let x0 = g.tmpc_to_pix(37.0, 1000.0);
        let x1 = g.tmpc_to_pix(41.0, 1000.0);
        let lvls = [
            (pcl.p0c, pcl.hght0c, "0 C"),
            (pcl.pm20c, pcl.hghtm20c, "-20 C"),
            (pcl.pm30c, pcl.hghtm30c, "-30 C"),
        ];
        for (p, h, label) in lvls {
            if !qc(p) {
                continue;
            }
            let y = g.pres_to_pix(p);
            dp.line_segment(
                [g.pt(x0, y), g.pt(x1, y)],
                Stroke::new(2.0, st.sig_temp_level_color),
            );
            dp.text(
                g.pt(x0, y - 12.0),
                Align2::LEFT_TOP,
                format!("{}={}'", label, int2str(m2ft(h))),
                fonts.hght.clone(),
                st.sig_temp_level_color,
            );
        }
    }

    fn draw_max_lapse_rate_layer(&self, dp: &Painter, painter: &Painter, g: &Geom, fonts: &Fonts) {
        let st = &self.style;
        let prof = self.prof;
        let (lr, pbot, ptop) = prof.max_lapse_rate_2_6;
        if !qc(pbot) || !qc(ptop) || !(lr >= 4.5) {
            return;
        }
        let x1 = g.tmpc_to_pix(
            prof.inner.interp_by_pressure(&prof.inner.vtmp, pbot) + 5.0,
            pbot,
        );
        let y1 = g.pres_to_pix(pbot);
        let y2 = g.pres_to_pix(ptop);
        dp.rect_filled(
            g.local_rect(x1 - 15.0, y2 - fonts.esrh_height, 50.0, fonts.esrh_height),
            0.0,
            st.bg_color,
        );
        let color = if lr >= 8.0 {
            st.alert_colors[5]
        } else if lr >= 7.0 {
            st.alert_colors[4]
        } else if lr >= 6.0 {
            st.alert_colors[1]
        } else {
            st.alert_colors[0]
        };
        let stroke = Stroke::new(1.5, color);
        dp.line_segment([g.pt(x1 - 10.0, y1), g.pt(x1 + 10.0, y1)], stroke);
        dp.line_segment([g.pt(x1 - 10.0, y2), g.pt(x1 + 10.0, y2)], stroke);
        dp.line_segment([g.pt(x1, y1), g.pt(x1, y2)], stroke);
        // Text drawn unclipped in the original.
        painter.text(
            g.pt(x1 - 15.0, y2 - fonts.esrh_height),
            Align2::LEFT_TOP,
            format!("{} C/km", crate::utils::float2str(lr, 1)),
            fonts.esrh.clone(),
            color,
        );
    }

    fn draw_effective_layer(&self, dp: &Painter, g: &Geom, fonts: &Fonts) {
        let st = &self.style;
        let prof = self.prof;
        let (pbot, ptop) = (prof.ebottom, prof.etop);
        if !qc(pbot) || !qc(ptop) {
            return;
        }
        let x1 = g.tmpc_to_pix(-20.0, 1000.0);
        let x2 = g.tmpc_to_pix(-33.0, 1000.0);
        let y1 = g.pres_to_pix(pbot);
        let y2 = g.pres_to_pix(ptop);
        let eh = fonts.esrh_height;
        let sfc = prof.inner.interp_hght(prof.inner.pres[prof.inner.sfc]);
        let text_bot = if prof.inner.pres[prof.inner.sfc] == pbot {
            "SFC".to_string()
        } else {
            format!("{}m", int2str(prof.inner.interp_hght(pbot) - sfc))
        };
        let text_top = format!("{}m", int2str(prof.inner.interp_hght(ptop) - sfc));
        dp.rect_filled(g.local_rect(x2, y1 + 4.0, 25.0, eh), 0.0, st.bg_color);
        dp.rect_filled(g.local_rect(x2, y2 - eh, 50.0, eh), 0.0, st.bg_color);
        dp.rect_filled(g.local_rect(x1 - 15.0, y2 - eh, 50.0, eh), 0.0, st.bg_color);
        let stroke = Stroke::new(2.0, st.eff_layer_color);
        let len = 15.0;
        dp.line_segment([g.pt(x1 - len, y1), g.pt(x1 + len, y1)], stroke);
        dp.line_segment([g.pt(x1 - len, y2), g.pt(x1 + len, y2)], stroke);
        dp.line_segment([g.pt(x1, y1), g.pt(x1, y2)], stroke);
        dp.text(
            g.pt(x2, y1 + 4.0),
            Align2::LEFT_TOP,
            text_bot,
            fonts.esrh.clone(),
            st.eff_layer_color,
        );
        dp.text(
            g.pt(x2, y2 - eh),
            Align2::LEFT_TOP,
            text_top,
            fonts.esrh.clone(),
            st.eff_layer_color,
        );
        dp.text(
            g.pt(x1 - 15.0, y2 - eh),
            Align2::LEFT_TOP,
            format!("{} m2s2", int2str(prof.right_esrh)),
            fonts.esrh.clone(),
            st.eff_layer_color,
        );
    }

    fn draw_barbs(&self, bp: &Painter, g: &Geom) {
        let prof = &self.prof.inner;
        let shemis = self.prof.latitude() < 0.0;
        if self.interp_winds {
            let mut p = prof.pres[prof.sfc];
            let ptop = prof.pres[prof.top];
            while p > ptop {
                let (wdir, wspd) = prof.interp_vec(p);
                if qc(wdir) && !wspd.is_nan() && p >= g.pmin {
                    let y = g.pres_to_pix(p);
                    draw_barb(bp, g.pt(g.barbx, y), wdir, wspd, shemis);
                }
                p -= 40.0;
            }
        } else {
            for i in 0..prof.pres.len() {
                if qc(prof.wdir[i]) && qc(prof.wspd[i]) && prof.pres[i] >= g.pmin {
                    let y = g.pres_to_pix(prof.pres[i]);
                    draw_barb(bp, g.pt(g.barbx, y), prof.wdir[i], prof.wspd[i], shemis);
                }
            }
        }
    }

    fn omeg_to_pix(&self, g: &Geom, omeg: f64) -> f64 {
        let plus10 = -49.0;
        let minus10 = -41.0;
        let x1_m10 = g.tmpc_to_pix(minus10, 1000.0);
        let x1_10 = g.tmpc_to_pix(plus10, 1000.0);
        let x1_0 = g.tmpc_to_pix((plus10 + minus10) / 2.0, 1000.0);
        if omeg > 0.0 {
            ((x1_0 - x1_10) / (0.0 - 10.0)) * omeg + x1_0
        } else if omeg < 0.0 {
            ((x1_0 - x1_m10) / (0.0 + 10.0)) * omeg + x1_0
        } else {
            x1_0
        }
    }

    fn draw_omega_profile(&self, dp: &Painter, g: &Geom, fonts: &Fonts) {
        let st = &self.style;
        let prof = self.prof;
        let plus10 = -49.0;
        let minus10 = -41.0;
        let x1_m10 = g.tmpc_to_pix(minus10, 1000.0);
        let x1_10 = g.tmpc_to_pix(plus10, 1000.0);
        let x1_0 = g.tmpc_to_pix((plus10 + minus10) / 2.0, 1000.0);
        let y1 = g.pres_to_pix(1000.0);
        let y2 = g.pres_to_pix(111.0);
        let frame = Stroke::new(1.0, st.omega_frame_color);
        // Dash-dot side rails (Qt's DashDotLine pattern: 4-2-1-2).
        for x in [x1_m10, x1_10] {
            let mut y = y2.min(y1);
            let yend = y2.max(y1);
            while y < yend {
                let d1 = (y + 4.0).min(yend);
                dp.line_segment([g.pt(x, y), g.pt(x, d1)], frame);
                let dot = y + 6.0;
                if dot < yend {
                    dp.line_segment([g.pt(x, dot), g.pt(x, (dot + 1.0).min(yend))], frame);
                }
                y += 9.0;
            }
        }
        dp.line_segment([g.pt(x1_0, y1), g.pt(x1_0, y2)], frame);
        dp.text(
            g.pt((x1_10 + x1_m10) / 2.0, y2 - 16.0),
            Align2::CENTER_CENTER,
            "OMEGA",
            fonts.esrh.clone(),
            st.omega_frame_color,
        );
        dp.text(
            g.pt(x1_m10, y2 - 5.0),
            Align2::CENTER_CENTER,
            "-10",
            fonts.esrh.clone(),
            st.omega_frame_color,
        );
        dp.text(
            g.pt(x1_10, y2 - 5.0),
            Align2::CENTER_CENTER,
            "+10",
            fonts.esrh.clone(),
            st.omega_frame_color,
        );

        for i in 0..prof.inner.omeg.len() {
            if !qc(prof.inner.omeg[i])
                || !prof.inner.pres[i].is_finite()
                || prof.inner.pres[i] < 111.0
            {
                continue;
            }
            let y = g.pres_to_pix(prof.inner.pres[i]);
            let (stroke, x2) = if prof.inner.omeg[i] > 0.0 {
                (
                    Stroke::new(1.5, st.omega_down_color),
                    self.omeg_to_pix(g, prof.inner.omeg[i] * 10.0),
                )
            } else if prof.inner.omeg[i] < 0.0 {
                (
                    Stroke::new(1.5, st.omega_up_color),
                    self.omeg_to_pix(g, prof.inner.omeg[i] * 10.0),
                )
            } else {
                (Stroke::new(1.0, st.omega_frame_color), x1_0)
            };
            dp.line_segment([g.pt(x1_0, y), g.pt(x2, y)], stroke);
        }
    }

    /// CAPE/CIN buoyancy fill (port of `sharpmod.viz.skew.draw_cape_fill`).
    fn draw_cape_fill(&self, fp: &Painter, g: &Geom, pcl: &ParcelResult) {
        let st = &self.style;
        let prof = self.prof;
        if pcl.ttrace.is_empty() {
            return;
        }
        // Finite parcel pairs.
        let mut p_tv = Vec::new();
        let mut p_p = Vec::new();
        for i in 0..pcl.ttrace.len().min(pcl.ptrace.len()) {
            if pcl.ttrace[i].is_finite() && pcl.ptrace[i].is_finite() {
                p_tv.push(pcl.ttrace[i]);
                p_p.push(pcl.ptrace[i]);
            }
        }
        // Finite environment pairs sorted by ascending pressure for interp.
        let mut env: Vec<(f64, f64)> = Vec::new();
        for i in 0..prof.inner.pres.len() {
            if prof.inner.vtmp[i].is_finite() && prof.inner.pres[i].is_finite() {
                env.push((prof.inner.pres[i], prof.inner.vtmp[i]));
            }
        }
        if p_tv.len() < 2 || env.len() < 2 {
            return;
        }
        env.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let interp_env = |p: f64| -> f64 {
            if p < env[0].0 || p > env[env.len() - 1].0 {
                return f64::NAN;
            }
            let mut lo = 0;
            let mut hi = env.len() - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if env[mid].0 <= p {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let (xa, ya) = env[lo];
            let (xb, yb) = env[hi];
            if xb == xa {
                ya
            } else {
                ya + (p - xa) / (xb - xa) * (yb - ya)
            }
        };

        let n = p_p.len();
        let mut env_on_parcel = vec![f64::NAN; n];
        for k in 0..n {
            env_on_parcel[k] = interp_env(p_p[k]);
        }
        let fill = |pts: &[(f64, f64)], color: Color32| {
            let poly: Vec<Pos2> = pts.iter().map(|(x, y)| g.pt(*x, *y)).collect();
            fp.add(Shape::convex_polygon(poly, color, Stroke::NONE));
        };
        for k in 0..n - 1 {
            if !env_on_parcel[k].is_finite() || !env_on_parcel[k + 1].is_finite() {
                continue;
            }
            let d0 = p_tv[k] - env_on_parcel[k];
            let d1 = p_tv[k + 1] - env_on_parcel[k + 1];
            if d0 == 0.0 && d1 == 0.0 {
                continue;
            }
            let y0 = g.pres_to_pix(p_p[k]);
            let y1 = g.pres_to_pix(p_p[k + 1]);
            let xp0 = g.tmpc_to_pix(p_tv[k], p_p[k]);
            let xp1 = g.tmpc_to_pix(p_tv[k + 1], p_p[k + 1]);
            let xe0 = g.tmpc_to_pix(env_on_parcel[k], p_p[k]);
            let xe1 = g.tmpc_to_pix(env_on_parcel[k + 1], p_p[k + 1]);
            if d0 * d1 >= 0.0 {
                let color = if d0 + d1 > 0.0 {
                    st.cape_fill_color
                } else {
                    st.cin_fill_color
                };
                fill(&[(xp0, y0), (xp1, y1), (xe1, y1), (xe0, y0)], color);
            } else {
                let f = d0 / (d0 - d1);
                let ym = y0 + f * (y1 - y0);
                let xm = xp0 + f * (xp1 - xp0);
                fill(
                    &[(xp0, y0), (xm, ym), (xe0, y0)],
                    if d0 > 0.0 {
                        st.cape_fill_color
                    } else {
                        st.cin_fill_color
                    },
                );
                fill(
                    &[(xm, ym), (xp1, y1), (xe1, y1)],
                    if d1 > 0.0 {
                        st.cape_fill_color
                    } else {
                        st.cin_fill_color
                    },
                );
            }
        }
    }

    /// HGZ overlay: dashed boundaries at the -10/-30 C isotherm levels plus
    /// the "HGZ" label (fill suppressed, matching the mounted configuration).
    fn draw_hgz(&self, painter: &Painter, g: &Geom, fonts: &Fonts) {
        let st = &self.style;
        let prof = self.prof;
        let p_warm =
            sharprs::params::indices::temp_lvl(&prof.inner, -10.0, false).unwrap_or(f64::NAN);
        let p_cold =
            sharprs::params::indices::temp_lvl(&prof.inner, -30.0, false).unwrap_or(f64::NAN);
        if !qc(p_warm) || !qc(p_cold) {
            return;
        }
        let mut left = g.rpad;
        if self.plot_omega_on() {
            let omega_right = g.tmpc_to_pix(-39.0, 1000.0);
            if omega_right.is_finite() {
                left = left.max(omega_right);
            }
        }
        let rect = Rect::from_min_max(g.pt(left, g.tpad), g.pt(g.brx, g.bry));
        let hp = painter.with_clip_rect(rect);
        let y_warm = g.pres_to_pix(p_warm);
        let y_cold = g.pres_to_pix(p_cold);
        let top = y_warm.min(y_cold).max(g.tpad);
        let bottom = y_warm.max(y_cold).min(g.bry);
        if bottom <= top {
            return;
        }
        let stroke = Stroke::new(1.0, st.hgz_edge_color);
        hp.extend(Shape::dashed_line(
            &[g.pt(left, top), g.pt(g.brx, top)],
            stroke,
            4.0,
            2.0,
        ));
        hp.extend(Shape::dashed_line(
            &[g.pt(left, bottom), g.pt(g.brx, bottom)],
            stroke,
            4.0,
            2.0,
        ));
        hp.text(
            g.pt(left + 4.0, top + 2.0),
            Align2::LEFT_TOP,
            "HGZ",
            fonts.hgz.clone(),
            st.hgz_edge_color,
        );
    }
}

/// The pressure (hPa) under `pos` when it lies inside the plot area of a
/// skew-T drawn into `rect` — shared by the widget's own readout and by
/// [`crate::SoundingView`]'s linked hodograph cursor.
pub(crate) fn hover_pressure(rect: Rect, pos: Pos2) -> Option<f64> {
    let g = Geom::new(rect);
    let x = (pos.x - rect.min.x) as f64;
    let y = (pos.y - rect.min.y) as f64;
    if x < g.lpad || x > g.wid + g.rpad || y <= g.tpad || y >= g.bry {
        return None;
    }
    let pres = g.pix_to_pres(y);
    if pres.is_finite() && pres <= g.pmax && pres >= g.pmin {
        Some(pres)
    } else {
        None
    }
}

impl SkewT<'_> {
    /// Hover readout (port of `plotSkewT.updateReadout`): a rubber-band line
    /// across the plot with height/pressure on the left and T/Td on the
    /// right, all interpolated to the cursor's pressure.
    fn draw_readout(&self, painter: &Painter, g: &Geom, pres: f64) {
        let st = &self.style;
        let inner = &self.prof.inner;
        let y = g.pres_to_pix(pres);

        painter.extend(Shape::dashed_line(
            &[g.pt(g.lpad, y), g.pt(g.brx, y)],
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 160)),
            3.0,
            3.0,
        ));

        let hgt_agl = inner.to_agl(inner.interp_hght(pres));
        let t = inner.interp_tmpc(pres);
        let td = inner.interp_dwpc(pres);
        let font = st.regular_font(11.0);

        let boxed = |pos: Pos2, anchor: Align2, text: String, color: Color32| {
            let galley = painter.layout_no_wrap(text.clone(), font.clone(), color);
            let r = anchor.anchor_size(pos, galley.size() + Vec2::new(6.0, 2.0));
            painter.rect_filled(r, 0.0, st.bg_color);
            painter.galley(r.min + Vec2::new(3.0, 1.0), galley, color);
        };
        // Left of the line: height AGL above, pressure below.
        boxed(
            g.pt(g.lpad + 1.0, y - 2.0),
            Align2::LEFT_BOTTOM,
            format!("{} m", crate::utils::float2str(hgt_agl, 1)),
            st.hgt_color,
        );
        boxed(
            g.pt(g.lpad + 1.0, y + 2.0),
            Align2::LEFT_TOP,
            format!("{} hPa", crate::utils::float2str(pres, 1)),
            st.fg_color,
        );
        // Right of the line: T above, Td below.
        boxed(
            g.pt(g.brx - 1.0, y - 2.0),
            Align2::RIGHT_BOTTOM,
            format!("T={} C", crate::utils::float2str(t, 1)),
            st.temp_color,
        );
        boxed(
            g.pt(g.brx - 1.0, y + 2.0),
            Align2::RIGHT_TOP,
            format!("Td={} C", crate::utils::float2str(td, 1)),
            st.dewp_color,
        );
    }
}

#[cfg(test)]
mod typography_tests {
    use super::*;

    #[test]
    fn text_scale_and_family_reach_the_skewt_font_bundle() {
        let base = Fonts::new(900.0, &SkewTStyle::default());
        let style = SkewTStyle::default()
            .with_font_preset(SoundingFontPreset::TechnicalMonospace)
            .with_text_scale(1.25);
        let scaled = Fonts::new(900.0, &style);

        assert!((scaled.title.size - base.title.size * 1.25).abs() < 0.001);
        assert!((scaled.in_plot.size - base.in_plot.size * 1.25).abs() < 0.001);
        assert_eq!(scaled.title.family, FontFamily::Monospace);
        assert_eq!(scaled.label_bold.family, FontFamily::Monospace);
        assert_eq!(scaled.title_height, base.title_height);
    }

    #[test]
    fn font_presets_are_cross_platform_egui_or_bundled_families() {
        let space = SkewTStyle::default().with_font_preset(SoundingFontPreset::SpaceGrotesk);
        assert_eq!(
            space.regular_font(10.0).family,
            FontFamily::Name(crate::FONT_FAMILY.into())
        );
        assert_eq!(
            space.bold_font(10.0).family,
            FontFamily::Name(crate::FONT_FAMILY_BOLD.into())
        );

        let clean = SkewTStyle::default()
            .with_font_preset(SoundingFontPreset::CleanProportional)
            .regular_font(10.0);
        assert_eq!(clean.family, FontFamily::Proportional);
    }
}
