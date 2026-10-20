//! Query-string settings (names per the public sounding API, plus a few extras)
//! parsed into typed sharppyrs / sharprs options, and the /api/options
//! schema the site builds its settings drawer from.

use std::collections::HashMap;

use egui::Color32;
use serde_json::{Value, json};
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
    pub brand: String,
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
        brand: qs(q, "brand").unwrap_or("gpuwm · sharppyrs").to_string(),
        style,
        blank,
        spag_t: spag.split(',').any(|s| s == "t"),
        spag_td: spag.split(',').any(|s| s == "td"),
        spag_alpha: qf(q, "spag_alpha")?.unwrap_or(110.0).clamp(10.0, 255.0) as u8,
    })
}

fn ch(v: &str, label: &str) -> Value {
    json!({"v": v, "label": label})
}

/// /api/options: every setting, its group, type, default and choices.
pub fn schema() -> Value {
    let d = SoundingLayout::default();
    let panels: Vec<Value> = PanelKind::ALL
        .iter()
        .map(|k| ch(k.token(), k.label()))
        .collect();
    let mut settings = vec![
        json!({"key":"parcel","label":"Lifted parcel","group":"Parcel","type":"enum","default":"mu",
            "choices":[ch("mu","Most unstable"),ch("ml","Mixed layer"),ch("sb","Surface"),ch("fcst","Forecast surface"),ch("user","User defined")]}),
        json!({"key":"mu_depth","label":"MU search depth","group":"Parcel","type":"number","min":50,"max":500,"step":10,"default":300,"unit":"hPa"}),
        json!({"key":"ml_depth","label":"Mixed-layer depth","group":"Parcel","type":"number","min":25,"max":300,"step":5,"default":100,"unit":"hPa"}),
        json!({"key":"user_p","label":"User parcel pressure","group":"Parcel","type":"number","min":100,"max":1100,"step":1,"default":null,"unit":"hPa","show_if":{"parcel":["user"]},"placeholder":"surface"}),
        json!({"key":"user_t","label":"User parcel temperature","group":"Parcel","type":"number","min":-80,"max":50,"step":0.5,"default":null,"unit":"C","show_if":{"parcel":["user"]},"placeholder":"surface"}),
        json!({"key":"user_td","label":"User parcel dewpoint","group":"Parcel","type":"number","min":-90,"max":40,"step":0.5,"default":null,"unit":"C","show_if":{"parcel":["user"]},"placeholder":"surface"}),
        json!({"key":"sm","label":"Storm motion","group":"Storm motion","type":"enum","default":"bunkers_rm",
            "choices":[ch("bunkers_rm","Bunkers right"),ch("bunkers_lm","Bunkers left"),ch("mean_wind","0-6 km mean wind"),
                       ch("mean_wind_npw","0-6 km mean wind (non-pressure-weighted)"),ch("corfidi_up","Corfidi upshear"),
                       ch("corfidi_down","Corfidi downshear"),ch("bunkers_np","Bunkers right (non-parcel)"),ch("user","User")]}),
        json!({"key":"sm_dir","label":"Storm motion from","group":"Storm motion","type":"number","min":0,"max":360,"step":5,"default":null,"unit":"deg","show_if":{"sm":["user"]}}),
        json!({"key":"sm_spd","label":"Storm motion speed","group":"Storm motion","type":"number","min":0,"max":150,"step":1,"default":null,"unit":"kt","show_if":{"sm":["user"]}}),
        json!({"key":"layout","label":"Window layout","group":"Layout","type":"layout",
            "slots":{"strips":2,"main":1,"insets":4,"bottom":6},"panels":panels,"default":d.to_tokens()}),
        json!({"key":"hodo_zoom","label":"Hodograph width","group":"Layout","type":"number","min":80,"max":500,"step":10,"default":d.hodo_zoom_kts,"unit":"kt"}),
        json!({"key":"corner","label":"Corner panel","group":"Layout","type":"enum","default":"locationmap",
            "choices":[ch("locationmap","Location map"),ch("hazardtype","Possible hazard type")]}),
        json!({"key":"view","label":"View","group":"Layout","type":"enum","default":"window",
            "choices":[ch("window","Full SPC window"),ch("skewt","Skew-T only")]}),
        json!({"key":"top_frac","label":"Upper row height","group":"Layout","type":"number","min":0.40,"max":0.85,"step":0.01,"default":d.top_height_fraction}),
        json!({"key":"skew_frac","label":"Skew-T width","group":"Layout","type":"number","min":0.30,"max":0.70,"step":0.01,"default":d.skew_width_fraction}),
        json!({"key":"omega","label":"Omega meter","group":"Skew-T","type":"bool","default":true,"doc":"default: on when the model provides omega"}),
        json!({"key":"interp_winds","label":"Barbs every 40 hPa","group":"Skew-T","type":"bool","default":true,"doc":"off = barbs at model levels"}),
        json!({"key":"font","label":"Font","group":"Text","type":"enum","default":"space_grotesk",
            "choices":[ch("space_grotesk","Space Grotesk"),ch("clean_proportional","Clean sans"),ch("technical_monospace","Technical mono")]}),
        json!({"key":"text_scale","label":"Text size","group":"Text","type":"number","min":0.5,"max":2.0,"step":0.05,"default":1.0}),
        json!({"key":"title","label":"Title","group":"Text","type":"text","default":""}),
        json!({"key":"brand","label":"Brand","group":"Text","type":"text","default":"gpuwm · sharppyrs"}),
        json!({"key":"theme","label":"Theme","group":"Display","type":"enum","default":"dark","choices":[ch("dark","Dark (SPC)"),ch("light","Light")]}),
        json!({"key":"w","label":"Width","group":"Display","type":"number","min":400,"max":4000,"step":10,"default":1630,"unit":"px"}),
        json!({"key":"h","label":"Height","group":"Display","type":"number","min":300,"max":4000,"step":10,"default":1100,"unit":"px"}),
        json!({"key":"scale","label":"Pixel ratio","group":"Display","type":"enum","default":"1","choices":[ch("1","1x"),ch("2","2x")]}),
        json!({"key":"blank","label":"Hidden table cells","group":"Tables","type":"cells","cells":tables::cells_catalogue()}),
        json!({"key":"mode","label":"Ensemble view","group":"Ensemble","type":"enum","default":"spaghetti",
            "choices":[ch("spaghetti","Spaghetti (every member)"),ch("envelope","Envelope (p10-p90, p25-p75, median)")]}),
        json!({"key":"spag","label":"Member traces","group":"Ensemble","type":"enum","default":"t,td",
            "choices":[ch("t,td","T and Td"),ch("t","T only"),ch("td","Td only")]}),
        json!({"key":"spag_alpha","label":"Member trace opacity","group":"Ensemble","type":"number","min":10,"max":255,"step":5,"default":110}),
        json!({"key":"field","label":"Time-height field","group":"Time-height","type":"enum","default":"rh",
            "choices":[ch("rh","Relative humidity"),ch("t","Temperature"),ch("td","Dewpoint"),ch("thetae","Theta-e"),ch("wspd","Wind speed"),
                       ch("omega","Omega"),ch("srh","Cumulative SRH"),ch("tdd","Dewpoint depression"),ch("tw","Wet-bulb temperature")]}),
        json!({"key":"ztop","label":"Time-height top","group":"Time-height","type":"number","min":1,"max":16,"step":0.5,"default":10,"unit":"km AGL"}),
        json!({"key":"barbs","label":"Time-height wind barbs","group":"Time-height","type":"bool","default":true}),
    ];
    for c in COLOR_FIELDS {
        settings.push(json!({"key": format!("c_{c}"), "label": c.trim_end_matches("_color").replace('_', " "), "group": "Colours", "type": "text", "default": "", "placeholder": "RRGGBB"}));
    }
    json!({
        "version": 1,
        "settings": settings,
        "presets": [
            {"name": "SPC default", "settings": {}},
            {"name": "Mixed-layer parcel", "settings": {"parcel": "ml"}},
            {"name": "Skew-T only", "settings": {"view": "skewt", "w": 900, "h": 900}},
            {"name": "Light", "settings": {"theme": "light"}},
            {"name": "Big text", "settings": {"text_scale": 1.25, "scale": "2"}},
        ],
    })
}

/// Catalogue served at /api/sounding/meta (the first API shape, which the
/// site's settings drawer is built from). Same option names the parsers take.
pub fn legacy_catalogue() -> Value {
    let panels: Vec<&str> = PanelKind::ALL.iter().map(|k| k.token()).collect();
    let slots = |k| {
        sharppyrs::native_diagnostic_slot_ids(k)
            .iter()
            .filter(|s| !tables::BLOCKED.contains(s))
            .copied()
            .collect::<Vec<_>>()
    };
    let d = SoundingLayout::default();
    json!({
        "selection": {
            "cycle": {"type": "string", "default": "latest", "doc": "YYYYMMDDHHMM analysis cycle"},
            "lead": {"type": "int", "unit": "min", "default": "earliest available", "doc": "forecast lead in minutes (0 = analysis)"},
            "member": {"type": "enum", "choices": ["memNNN", "N", "mean", "all"], "default": "mean",
                       "doc": "one member, the ensemble mean, or all members (mean drawn + member spaghetti + band)"},
            "members": {"type": "csv", "doc": "restrict mean/all to these members"},
            "lat": {"type": "float"}, "lon": {"type": "float"},
            "bbox": {"type": "w,s,e,n", "endpoint": "box"},
            "poly": {"type": "lon,lat;lon,lat;...", "endpoint": "box"},
        },
        "analysis": {
            "parcel": {"type": "enum", "choices": ["sb", "fcst", "mu", "ml", "user"], "default": "mu", "doc": "parcel trace drawn on the skew-T"},
            "ml_depth": {"type": "float", "unit": "hPa", "default": 100, "range": [10, 500]},
            "mu_depth": {"type": "float", "unit": "hPa", "default": 300, "range": [10, 700]},
            "user_p": {"type": "float", "unit": "hPa"}, "user_t": {"type": "float", "unit": "C"},
            "user_td": {"type": "float", "unit": "C", "doc": "user parcel (missing values default to the surface); shown in the FCST row, select with parcel=user"},
            "sm": {"type": "enum", "choices": ["brm", "blm", "mean06", "mean06npw", "corfidi_up", "corfidi_down", "bunkers_np", "user"], "default": "brm",
                   "doc": "storm motion for SRH / SR winds / hodograph marker"},
            "sm_dir": {"type": "float", "unit": "deg from"}, "sm_spd": {"type": "float", "unit": "kt"},
        },
        "display": {
            "format": {"choices": ["png", "json"], "doc": "path suffix .png or .json"},
            "view": {"type": "enum", "choices": ["window", "skewt"], "default": "window", "doc": "full SPC window or the skew-T alone"},
            "w": {"type": "int", "default": "1630 (window) / 900 (skewt)", "range": [400, 4000]},
            "h": {"type": "int", "default": "1100 (window) / 900 (skewt)", "range": [300, 4000]},
            "scale": {"type": "float", "default": 1.0, "range": [0.5, 3.0], "doc": "pixels per point (2 = retina)"},
            "layout": {"type": "string", "default": d.to_tokens(), "doc": "sharppyrs SoundingLayout tokens (strips|main|insets|bottom|hodo_zoom[|g2:...])"},
            "strip1": {"choices": panels, "default": d.strips[0].token()}, "strip2": {"choices": panels, "default": d.strips[1].token()},
            "main": {"choices": panels, "default": d.main.token()},
            "inset1": {"choices": panels, "default": d.insets[0].token()}, "inset2": {"choices": panels, "default": d.insets[1].token()},
            "inset3": {"choices": panels, "default": d.insets[2].token()}, "inset4": {"choices": panels, "default": d.insets[3].token()},
            "bottom1": {"choices": panels, "default": d.bottom[0].token()}, "bottom2": {"choices": panels, "default": d.bottom[1].token()},
            "bottom3": {"choices": panels, "default": d.bottom[2].token()}, "bottom4": {"choices": panels, "default": d.bottom[3].token()},
            "bottom5": {"choices": panels, "default": d.bottom[4].token()}, "bottom6": {"choices": panels, "default": d.bottom[5].token()},
            "hodo_zoom": {"type": "float", "unit": "kt across", "default": d.hodo_zoom_kts, "range": [80, 500]},
            "top_frac": {"type": "float", "default": d.top_height_fraction, "range": [0.40, 0.85]},
            "skew_frac": {"type": "float", "default": d.skew_width_fraction, "range": [0.30, 0.70]},
            "corner": {"choices": ["locationmap", "hazardtype"], "default": "locationmap"},
            "font": {"choices": ["spacegrotesk", "clean", "mono"], "default": "spacegrotesk"},
            "text_scale": {"type": "float", "default": 1.0, "range": [0.5, 2.0]},
            "theme": {"choices": ["dark", "light"], "default": "dark"},
            "title": {"type": "string", "default": "auto"}, "brand": {"type": "string", "default": "gpuwm · sharppyrs"},
            "hide": {"type": "csv", "choices": {
                "convective": slots(DiagnosticTablePanelKind::Convective),
                "kinematics": slots(DiagnosticTablePanelKind::Kinematics),
                "severe": slots(DiagnosticTablePanelKind::Severe)}, "doc": "blank these table cells"},
            "colors": {"params": COLOR_FIELDS.iter().map(|c| format!("c_{c}")).collect::<Vec<_>>(), "type": "RRGGBB[AA]"},
            "omega": {"choices": ["auto", "on", "off"], "default": "auto", "doc": "skew-T omega meter"},
            "interp_winds": {"choices": ["on", "off"], "default": "on", "doc": "barbs every 40 hPa (on) or at model levels (off)"},
            "mode": {"choices": ["spaghetti", "envelope"], "default": "spaghetti", "doc": "member=all: every member's trace, or p10-p90 + p25-p75 bands with the median"},
            "spag": {"type": "csv", "choices": ["t", "td"], "default": "t,td", "doc": "member traces drawn when member=all"},
            "spag_alpha": {"type": "int", "default": 110, "range": [10, 255]},
            "envelope": {"choices": ["on", "off"], "default": "on", "doc": "member percentile band when member=all"},
        },
        "series": {
            "field": {"choices": ["rh", "thetae", "t", "td", "tw", "tdd", "wspd", "omega", "srh"], "default": "rh"},
            "ztop": {"type": "float", "unit": "km AGL", "default": 10, "range": [1, 16]},
            "barbs": {"choices": ["on", "off"], "default": "on"},
            "leads": {"type": "csv", "doc": "subset of leads (min); default every lead"},
        },
    })
}
