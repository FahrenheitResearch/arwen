//! Batch settings parsed into typed sharppyrs / sharprs options.

use std::collections::HashMap;

use egui::Color32;
use sharppyrs::{
    CornerPanel, DiagnosticTablePanelKind, PanelKind, ParcelType, SkewTStyle, SoundingFontPreset,
    SoundingLayout,
};

use crate::analysis::{AnalysisOpts, StormMotion};
use crate::render::ViewKind;
use crate::store::Res;
use crate::tables;

pub type Q = HashMap<String, String>;

pub fn qs<'a>(q: &'a Q, k: &str) -> Option<&'a str> {
    q.get(k).map(|s| s.as_str()).filter(|s| !s.is_empty())
}
pub fn qf(q: &Q, k: &str) -> Res<Option<f64>> {
    match qs(q, k) {
        None => Ok(None),
        Some(s) => s
            .parse::<f64>()
            .map(Some)
            .map_err(|_| format!("{k}: not a number '{s}'")),
    }
}
pub fn qb(q: &Q, k: &str, d: bool) -> Res<bool> {
    match qs(q, k) {
        None => Ok(d),
        Some("1" | "true" | "on" | "yes") => Ok(true),
        Some("0" | "false" | "off" | "no") => Ok(false),
        Some(s) => Err(format!("{k}: expected 0/1, got '{s}'")),
    }
}

pub struct DisplayOpts {
    pub view: ViewKind,
    pub width: f32,
    pub height: f32,
    pub ppp: f32,
    pub layout: SoundingLayout,
    pub corner: CornerPanel,
    pub parcel: ParcelType,
    pub title: Option<String>,
    /// The `brand` setting (top right), when the caller gave one. Unset, the label is built from the run's own
    /// description (batch.rs `brand_from`), never a tool name.
    pub brand: Option<String>,
    pub style: SkewTStyle,
    pub blank: Vec<(DiagnosticTablePanelKind, String)>,
    pub spag_t: bool,
    pub spag_td: bool,
    pub spag_alpha: u8,
}

fn parse_color(s: &str) -> Res<Color32> {
    let h = s.trim_start_matches('#');
    let p =
        |i: usize| u8::from_str_radix(&h[i..i + 2], 16).map_err(|_| format!("bad colour '{s}'"));
    match h.len() {
        6 => Ok(Color32::from_rgb(p(0)?, p(2)?, p(4)?)),
        8 => Ok(Color32::from_rgba_unmultiplied(p(0)?, p(2)?, p(4)?, p(6)?)),
        _ => Err(format!("bad colour '{s}' (want RRGGBB or RRGGBBAA)")),
    }
}

pub const COLOR_FIELDS: [&str; 22] = [
    "bg_color",
    "fg_color",
    "isotherm_color",
    "isotherm_hgz_color",
    "adiab_color",
    "mixr_color",
    "temp_color",
    "dewp_color",
    "wetbulb_color",
    "eff_layer_color",
    "hgt_color",
    "lcl_mkr_color",
    "lfc_mkr_color",
    "el_mkr_color",
    "sig_temp_level_color",
    "cape_fill_color",
    "cin_fill_color",
    "hgz_edge_color",
    "omega_up_color",
    "omega_down_color",
    "omega_frame_color",
    "downdraft_color",
];

fn color_field<'a>(st: &'a mut SkewTStyle, name: &str) -> Option<&'a mut Color32> {
    Some(match name {
        "bg_color" => &mut st.bg_color,
        "fg_color" => &mut st.fg_color,
        "isotherm_color" => &mut st.isotherm_color,
        "isotherm_hgz_color" => &mut st.isotherm_hgz_color,
        "adiab_color" => &mut st.adiab_color,
        "mixr_color" => &mut st.mixr_color,
        "temp_color" => &mut st.temp_color,
        "dewp_color" => &mut st.dewp_color,
        "wetbulb_color" => &mut st.wetbulb_color,
        "eff_layer_color" => &mut st.eff_layer_color,
        "hgt_color" => &mut st.hgt_color,
        "lcl_mkr_color" => &mut st.lcl_mkr_color,
        "lfc_mkr_color" => &mut st.lfc_mkr_color,
        "el_mkr_color" => &mut st.el_mkr_color,
        "sig_temp_level_color" => &mut st.sig_temp_level_color,
        "cape_fill_color" => &mut st.cape_fill_color,
        "cin_fill_color" => &mut st.cin_fill_color,
        "hgz_edge_color" => &mut st.hgz_edge_color,
        "omega_up_color" => &mut st.omega_up_color,
        "omega_down_color" => &mut st.omega_down_color,
        "omega_frame_color" => &mut st.omega_frame_color,
        "downdraft_color" => &mut st.downdraft_color,
        _ => return None,
    })
}

/// Light theme: the SkewTStyle colours re-picked for a white page. Panels
/// that draw with fixed colours keep them.
fn light(mut st: SkewTStyle) -> SkewTStyle {
    let c = Color32::from_rgb;
    st.bg_color = c(255, 255, 255);
    st.fg_color = c(20, 20, 20);
    st.isotherm_color = c(170, 170, 170);
    st.isotherm_hgz_color = c(40, 90, 200);
    st.adiab_color = c(150, 150, 150);
    st.mixr_color = c(60, 140, 60);
    st.temp_color = c(210, 20, 20);
    st.dewp_color = c(0, 150, 40);
    st.wetbulb_color = c(0, 150, 200);
    st.hgt_color = c(200, 30, 30);
    st.eff_layer_color = c(0, 140, 170);
    st.cape_fill_color = Color32::from_rgba_unmultiplied(255, 140, 40, 90);
    st.cin_fill_color = Color32::from_rgba_unmultiplied(60, 120, 255, 70);
    st.omega_frame_color = c(120, 120, 120);
    st
}

fn panel_kind(s: &str) -> Res<PanelKind> {
    PanelKind::from_token(s).ok_or_else(|| format!("unknown panel '{s}'"))
}

pub fn parse_analysis(q: &Q) -> Res<AnalysisOpts> {
    let mut o = AnalysisOpts::default();
    if let Some(v) = qf(q, "ml_depth")? {
        if !(10.0..=500.0).contains(&v) {
            return Err("ml_depth must be 10..500 hPa".into());
        }
        o.ml_depth_hpa = v;
    }
    if let Some(v) = qf(q, "mu_depth")? {
        if !(10.0..=700.0).contains(&v) {
            return Err("mu_depth must be 10..700 hPa".into());
        }
        o.mu_depth_hpa = v;
    }
    let (up, ut, utd) = (qf(q, "user_p")?, qf(q, "user_t")?, qf(q, "user_td")?);
    if qs(q, "parcel") == Some("user") || up.is_some() || ut.is_some() || utd.is_some() {
        o.user_parcel = Some((up, ut, utd));
    }
    o.storm_motion = StormMotion::parse(
        qs(q, "sm").unwrap_or(""),
        qf(q, "sm_dir")?,
        qf(q, "sm_spd")?,
    )?;
    Ok(o)
}

pub fn parse_display(q: &Q) -> Res<DisplayOpts> {
    let omega = match qs(q, "omega") {
        None | Some("auto") => None,
        Some(_) => Some(qb(q, "omega", true)?),
    };
    let interp_winds = qb(q, "interp_winds", true)?;
    let view = match qs(q, "view").unwrap_or("window") {
        "window" | "spc" => ViewKind::Window {
            omega,
            interp_winds,
        },
        "skewt" => ViewKind::SkewT {
            omega,
            interp_winds,
        },
        v => return Err(format!("view must be window|skewt, got '{v}'")),
    };
    let (dw, dh) = match view {
        ViewKind::Window { .. } => (1630.0, 1100.0),
        ViewKind::SkewT { .. } => (900.0, 900.0),
    };
    let width = qf(q, "w")?.unwrap_or(dw).clamp(400.0, 4000.0) as f32;
    let height = qf(q, "h")?.unwrap_or(dh).clamp(300.0, 4000.0) as f32;
    let ppp = qf(q, "scale")?.unwrap_or(1.0).clamp(0.5, 3.0) as f32;
    let mut layout = match qs(q, "layout") {
        Some(t) => {
            SoundingLayout::from_tokens(t).ok_or("layout: unparseable SoundingLayout tokens")?
        }
        None => SoundingLayout::default(),
    };
    for i in 0..2 {
        if let Some(s) = qs(q, &format!("strip{}", i + 1)) {
            layout.strips[i] = panel_kind(s)?;
        }
    }
    if let Some(s) = qs(q, "main") {
        layout.main = panel_kind(s)?;
    }
    for i in 0..4 {
        if let Some(s) = qs(q, &format!("inset{}", i + 1)) {
            layout.insets[i] = panel_kind(s)?;
        }
    }
    for i in 0..6 {
        if let Some(s) = qs(q, &format!("bottom{}", i + 1)) {
            layout.bottom[i] = panel_kind(s)?;
        }
    }
    if let Some(v) = qf(q, "top_frac")? {
        layout.top_height_fraction = v as f32;
    }
    if let Some(v) = qf(q, "skew_frac")? {
        layout.skew_width_fraction = v as f32;
    }
    layout = SoundingLayout::from_tokens(&layout.to_tokens()).unwrap_or(layout);
    if let Some(z) = qf(q, "hodo_zoom")? {
        layout.hodo_zoom_kts = z.clamp(80.0, 500.0);
    }
    let corner = match qs(q, "corner").unwrap_or("locationmap") {
        "locationmap" | "map" => CornerPanel::LocationMap,
        "hazardtype" | "hazard" => CornerPanel::HazardType,
        c => return Err(format!("corner must be locationmap|hazardtype, got '{c}'")),
    };
    if corner == CornerPanel::HazardType
        && qs(q, "inset4").is_none()
        && layout.insets[3] == PanelKind::LocationMap
    {
        layout.insets[3] = PanelKind::HazardType;
    }
    let parcel = match qs(q, "parcel").unwrap_or("mu") {
        "sb" | "sfc" | "surface" => ParcelType::Surface,
        "fcst" | "forecast" | "user" => ParcelType::Forecast,
        "mu" => ParcelType::MostUnstable,
        "ml" => ParcelType::MixedLayer,
        p => return Err(format!("parcel must be mu|ml|sb|fcst|user, got '{p}'")),
    };
    let preset = match qs(q, "font").unwrap_or("space_grotesk") {
        "space_grotesk" | "spacegrotesk" => SoundingFontPreset::SpaceGrotesk,
        "clean_proportional" | "clean" => SoundingFontPreset::CleanProportional,
        "technical_monospace" | "mono" => SoundingFontPreset::TechnicalMonospace,
        f => {
            return Err(format!(
                "font must be space_grotesk|clean_proportional|technical_monospace, got '{f}'"
            ));
        }
    };
    let mut style = SkewTStyle::space_grotesk()
        .with_font_preset(preset)
        .with_text_scale(qf(q, "text_scale")?.unwrap_or(1.0).clamp(0.5, 2.0) as f32);
    match qs(q, "theme").unwrap_or("dark") {
        "dark" => {}
        "light" => style = light(style),
        t => return Err(format!("theme must be dark|light, got '{t}'")),
    }
    for name in COLOR_FIELDS {
        if let Some(v) = qs(q, &format!("c_{name}")) {
            *color_field(&mut style, name).unwrap() = parse_color(v)?;
        }
    }
    let mut blank = vec![];
    for key in ["blank", "hide"] {
        if let Some(list) = qs(q, key) {
            for id in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                if tables::BLOCKED.contains(&id) {
                    continue; // always blank anyway
                }
                let k = tables::slot_panel(id)
                    .ok_or_else(|| format!("{key}: unknown table cell '{id}'"))?;
                blank.push((k, id.to_string()));
            }
        }
    }
    let spag = qs(q, "spag").unwrap_or("t,td");
    Ok(DisplayOpts {
        view,
        width,
        height,
        ppp,
        layout,
        corner,
        parcel,
        title: qs(q, "title").map(str::to_string),
        brand: qs(q, "brand").map(str::trim).filter(|s| !s.is_empty()).map(str::to_string),
        style,
        blank,
        spag_t: spag.split(',').any(|s| s == "t"),
        spag_td: spag.split(',').any(|s| s == "td"),
        spag_alpha: qf(q, "spag_alpha")?.unwrap_or(110.0).clamp(10.0, 255.0) as u8,
    })
}
