//! Run differences: one product drawn as run A minus run B, on the grid both
//! runs share, at the valid time both runs share.
//!
//! The difference is taken on the values the product itself draws.  Every
//! lane (direct fields, derived diagnostics, stored variables) ends at one
//! save call with a finished [`MapRenderRequest`]; this module sits in front
//! of that call.  A difference run renders run B first with the stage set to
//! [`DifferenceStage::Capture`]: each product's drawn field is kept instead
//! of written.  Run A then renders with the stage set to
//! [`DifferenceStage::Subtract`]: each product's field has run B's field of
//! the same product subtracted from it, cell by cell, and the result is
//! drawn in place of run A's picture.  So a derived product is computed from
//! each run's own inputs first and differenced after, and no product needs
//! code of its own to take part.
//!
//! Rules, each one because the alternative draws something untrue:
//!
//! * The two fields must be on one grid: the same shape and the same cell
//!   coordinates to within [`GRID_TOLERANCE_DEG`].  Anything else is refused
//!   by name.  Nothing is regridded, because a regrid invents values and a
//!   difference of invented values is a picture of the regrid.
//! * Units must match.  A product drawn in two units by two runs is refused.
//! * A product that draws values below a floor as nothing (reflectivity
//!   below its first level, CAPE below its mask) differences those values as
//!   the floor, and a cell below the floor in both runs is left undrawn.
//!   Otherwise every no-echo cell of both runs would be drawn as a large
//!   difference of two sentinels.
//! * An RGB product and a category product carry no quantity to subtract
//!   and are refused by name.  So is a sheet composed of several panels.
//! * Overlays (contours, barbs, streamlines) are left out: two runs'
//!   contours cannot be subtracted into one set of lines.
//! * The colour bar is symmetric about zero; its half range comes from
//!   `data/difference.json` (see [`half_range`]) and the rule that set it is
//!   reported with every panel.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde::Deserialize;

use crate::chrome_plan;
use crate::colormap::{LegendMode, LevelDensity, RenderDensity};
use crate::presentation::{ProductVisualMode, StaticPlotStyle};
use crate::render::PngWriteOptions;
use crate::request::{
    Color, ColorScale, DiscreteColorScale, ExtendMode, Field2D, LatLonGrid, MapRenderRequest,
    ProductKey,
};
use crate::{RenderSaveTiming, RustRenderer, RustwxRenderError};

/// Two grids whose cell coordinates differ by more than this many degrees
/// anywhere are two grids.  About 11 m: far below any model spacing, far
/// above the rounding of one grid written twice.
pub const GRID_TOLERANCE_DEG: f32 = 1.0e-4;

const TABLE_JSON: &str = include_str!("../data/difference.json");

#[derive(Debug, Clone, Deserialize)]
pub struct DifferenceTable {
    pub bands: usize,
    pub neutral_bands: usize,
    pub percentile: f64,
    pub colors: Vec<String>,
    pub rows: Vec<DifferenceRow>,
    /// Products whose title names more than the filled field in a way the
    /// ` / ` rule of [`difference_title`] cannot take apart.
    #[serde(default)]
    pub titles: Vec<DifferenceTitle>,
}

/// The name of the quantity a product's difference subtracts, for a
/// product whose own title also names what its overlays draw.
#[derive(Debug, Clone, Deserialize)]
pub struct DifferenceTitle {
    pub product: String,
    pub title: String,
}

/// True when `product_key` is `product` or ends with `_<product>`: the key
/// carries the domain in front of the product (`d01-3km_2m_temperature`).
fn key_names(product_key: &str, product: &str) -> bool {
    product_key == product
        || product_key
            .strip_suffix(product)
            .is_some_and(|head| head.ends_with('_'))
}

/// A fixed half range for one product in one unit, so every frame of a
/// loop shares one bar.
#[derive(Debug, Clone, Deserialize)]
pub struct DifferenceRow {
    pub product: String,
    pub units: String,
    pub half_range: f64,
}

impl DifferenceTable {
    pub fn builtin() -> &'static DifferenceTable {
        static TABLE: OnceLock<DifferenceTable> = OnceLock::new();
        TABLE.get_or_init(|| {
            let table: DifferenceTable = serde_json::from_str(TABLE_JSON)
                .expect("data/difference.json is compiled in and parsed by a unit test");
            table
        })
    }

    /// The row for `product_key` drawn in `units`: the longest row product
    /// that the key ends with, at a `_` boundary.  The key carries the
    /// domain in front of the product (`d01-3km_2m_temperature`).
    pub fn row_for(&self, product_key: &str, units: &str) -> Option<&DifferenceRow> {
        self.rows
            .iter()
            .filter(|row| row.units == units)
            .filter(|row| key_names(product_key, &row.product))
            .max_by_key(|row| row.product.len())
    }

    /// The table's title for `product_key`'s difference, if it has one.
    pub fn title_for(&self, product_key: &str) -> Option<&str> {
        self.titles
            .iter()
            .filter(|row| key_names(product_key, &row.product))
            .max_by_key(|row| row.product.len())
            .map(|row| row.title.as_str())
    }

    /// The band colours: the theme's diverging ramp when it names one,
    /// otherwise the table's own, with the bands nearest zero neutral.
    pub fn band_colors(&self) -> Vec<Color> {
        let bands = self.bands.max(2);
        let theme = crate::theme::active_theme();
        let mut colors = match theme.diverging_colors(bands) {
            Some(colors) => {
                let neutral = theme
                    .diverging_colors(3)
                    .map(|mid| mid[1])
                    .unwrap_or(Color::WHITE);
                let mut colors = colors;
                neutralize(&mut colors, self.neutral_bands, neutral);
                colors
            }
            None => {
                let anchors: Vec<crate::Rgba> = self
                    .colors
                    .iter()
                    .filter_map(|hex| crate::theme::parse_color(hex).ok())
                    .collect();
                if anchors.len() == bands {
                    anchors.into_iter().map(Color::from).collect()
                } else {
                    crate::theme::resample(&anchors, bands)
                }
            }
        };
        colors.truncate(bands);
        colors
    }
}

fn neutralize(colors: &mut [Color], neutral_bands: usize, neutral: Color) {
    let n = colors.len();
    let neutral_bands = neutral_bands.min(n);
    let start = (n - neutral_bands) / 2;
    for color in colors.iter_mut().skip(start).take(neutral_bands) {
        *color = neutral;
    }
}

/// How the half range of one difference bar was chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum RangeRule {
    /// The table row for the product in its units.
    Table,
    /// The table's percentile of |A - B| (`observed`), rounded up to a
    /// 1-2-5 band step.
    Percentile { percentile: f64, observed: f64 },
    /// No cell has a defined difference, or every one is zero.
    NoDifference,
}

impl RangeRule {
    pub fn describe(&self) -> String {
        match self {
            RangeRule::Table => "table".to_string(),
            RangeRule::Percentile {
                percentile,
                observed,
            } => format!("p{percentile}={observed:.4}"),
            RangeRule::NoDifference => "no-difference".to_string(),
        }
    }
}

/// The next 1-2-5 step at or above `value`; 1 for zero or less.
pub fn nice_step(value: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 {
        return 1.0;
    }
    let exponent = value.log10().floor();
    let base = 10f64.powf(exponent);
    for step in [1.0, 2.0, 5.0, 10.0] {
        if value <= step * base * 1.000_001 {
            return step * base;
        }
    }
    10.0 * base
}

/// The bar's half range and the rule that set it.
///
/// A table row wins.  Without one, the half range is the table's
/// percentile of |A - B| over the cells where the difference is defined,
/// rounded up so that one band is a 1, 2 or 5 step: with 10 bands a 99th
/// percentile of 7.3 gives bands of 2 and a half range of 10.
pub fn half_range(
    table: &DifferenceTable,
    product_key: &str,
    units: &str,
    values: &[f32],
) -> (f64, RangeRule) {
    let half_bands = (table.bands.max(2) / 2) as f64;
    if let Some(row) = table.row_for(product_key, units) {
        return (row.half_range, RangeRule::Table);
    }
    let mut magnitudes: Vec<f32> = values
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value.abs())
        .collect();
    if magnitudes.is_empty() {
        return (nice_step(0.0) * half_bands, RangeRule::NoDifference);
    }
    let rank = ((table.percentile / 100.0) * magnitudes.len() as f64).ceil() as usize;
    let index = rank.clamp(1, magnitudes.len()) - 1;
    let (_, observed, _) = magnitudes.select_nth_unstable_by(index, |a, b| a.total_cmp(b));
    let observed = f64::from(*observed);
    if observed <= 0.0 {
        return (nice_step(0.0) * half_bands, RangeRule::NoDifference);
    }
    (
        nice_step(observed / half_bands) * half_bands,
        RangeRule::Percentile {
            percentile: table.percentile,
            observed,
        },
    )
}

/// The zero-centred bar over `half_range`: `bands` equal bands from
/// `-half_range` to `+half_range`, zero on the middle edge, arrows both ways.
pub fn difference_scale(table: &DifferenceTable, half_range: f64) -> DiscreteColorScale {
    let bands = table.bands.max(2);
    let span = if half_range.is_finite() && half_range > 0.0 {
        half_range
    } else {
        1.0
    };
    let levels = (0..=bands)
        .map(|index| {
            let level = -span + 2.0 * span * index as f64 / bands as f64;
            // The middle edge is exactly zero, not -0 or 1e-16.
            if level.abs() < span * 1.0e-9 { 0.0 } else { level }
        })
        .collect();
    DiscreteColorScale {
        levels,
        colors: table.band_colors(),
        extend: ExtendMode::Both,
        mask_below: None,
    }
}

/// The value below which `scale` draws nothing, if it has one: its mask,
/// or its first level when it does not extend below it.
pub fn undrawn_floor(scale: &ColorScale) -> Option<f64> {
    let discrete = scale.resolved_discrete();
    if let Some(mask) = discrete.mask_below {
        return Some(mask);
    }
    match discrete.extend {
        ExtendMode::Neither | ExtendMode::Max => discrete.levels.first().copied(),
        ExtendMode::Min | ExtendMode::Both => None,
    }
}

/// Refuse two grids that are not one grid.
pub fn check_same_grid(a: &LatLonGrid, b: &LatLonGrid) -> Result<(), RustwxRenderError> {
    if a.shape != b.shape {
        return Err(RustwxRenderError::Difference(format!(
            "run A is on a {}x{} grid and run B on a {}x{} grid; a difference is taken on one \
             native grid and is never regridded",
            a.shape.nx, a.shape.ny, b.shape.nx, b.shape.ny
        )));
    }
    let mut worst = 0.0f32;
    let mut worst_cell = 0usize;
    for (index, ((lat_a, lat_b), (lon_a, lon_b))) in a
        .lat_deg
        .iter()
        .zip(&b.lat_deg)
        .zip(a.lon_deg.iter().zip(&b.lon_deg))
        .enumerate()
    {
        let dlon = (lon_a - lon_b).abs();
        let dlon = dlon.min((360.0 - dlon).abs());
        let off = (lat_a - lat_b).abs().max(dlon);
        if off > worst || off.is_nan() {
            worst = if off.is_nan() { f32::INFINITY } else { off };
            worst_cell = index;
        }
    }
    if worst > GRID_TOLERANCE_DEG {
        let nx = a.shape.nx.max(1);
        return Err(RustwxRenderError::Difference(format!(
            "run A and run B are both {}x{} but are not one grid: their cell coordinates differ \
             by {worst:.4} degrees at column {} row {}; a difference is taken on one native grid \
             and is never regridded",
            a.shape.nx,
            a.shape.ny,
            worst_cell % nx,
            worst_cell / nx
        )));
    }
    Ok(())
}

/// A minus B, cell by cell, on the one grid both are on.
///
/// With a `floor`, a value below it is taken as the floor, and a cell below
/// it in both runs is NaN (undrawn).  A NaN in either run is NaN.
pub fn subtract_fields(
    a: &Field2D,
    b: &Field2D,
    floor: Option<f64>,
) -> Result<Vec<f32>, RustwxRenderError> {
    check_same_grid(&a.grid, &b.grid)?;
    if a.units.trim() != b.units.trim() {
        return Err(RustwxRenderError::Difference(format!(
            "run A draws this product in {:?} and run B in {:?}; a difference needs one unit",
            a.units, b.units
        )));
    }
    let floor = floor.map(|value| value as f32);
    Ok(a.values
        .iter()
        .zip(&b.values)
        .map(|(&va, &vb)| {
            if va.is_nan() || vb.is_nan() {
                return f32::NAN;
            }
            match floor {
                Some(floor) if va < floor && vb < floor => f32::NAN,
                Some(floor) => va.max(floor) - vb.max(floor),
                None => va - vb,
            }
        })
        .collect())
}

/// What run B's pass kept of one product.
#[derive(Debug, Clone)]
pub struct CapturedPanel {
    pub field: Field2D,
    pub scale: ColorScale,
    pub categories: bool,
    pub rgb: bool,
    pub title: Option<String>,
    pub subtitle_left: Option<String>,
    /// Run B's own picture, when the caller asked for the run panels.
    pub panel: Option<PathBuf>,
}

impl CapturedPanel {
    pub fn from_request(request: &MapRenderRequest, panel: Option<PathBuf>) -> Self {
        Self {
            field: request.field.clone(),
            scale: request.scale.clone(),
            categories: request.legend.mode == LegendMode::Categories,
            rgb: request.rgba_grid.is_some(),
            title: request.title.clone(),
            subtitle_left: request.subtitle_left.clone(),
            panel,
        }
    }
}

/// The two runs' names, as the metadata line prints them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DifferenceLabels {
    pub a: String,
    pub b: String,
}

impl Default for DifferenceLabels {
    fn default() -> Self {
        Self {
            a: "A".to_string(),
            b: "B".to_string(),
        }
    }
}

/// One drawn difference, as the caller reports it.
#[derive(Debug, Clone)]
pub struct DrawnDifference {
    pub key: String,
    pub output: PathBuf,
    pub a_panel: Option<PathBuf>,
    pub b_panel: Option<PathBuf>,
    pub units: String,
    pub half_range: f64,
    pub step: f64,
    pub rule: RangeRule,
    /// Cells with a defined difference, and the largest |A - B| among them.
    pub defined_cells: usize,
    pub max_abs: f64,
}

fn split_tokens(text: Option<&str>) -> Vec<String> {
    text.map(|text| {
        text.split(" | ")
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

/// The difference panel's title: `<product> difference (<units>), A minus B`,
/// with the grid label kept at the end so the header moves it to the
/// metadata row as it does on every map.
///
/// Only the filled field is differenced; the overlays are dropped.  So a
/// product that drew overlays is named by its filled field alone: the
/// table's title row when it has one, otherwise the title up to its first
/// ` / ` (`MSLP / 10m Winds` differences MSLP).  Without this the title of
/// a wind-speed difference said it differenced the wind direction too.
pub fn difference_title(
    a_title: Option<&str>,
    product_key: &str,
    units: &str,
    had_overlays: bool,
    table: &DifferenceTable,
) -> String {
    let (base, domain) = chrome_plan::split_domain_label(a_title.unwrap_or(""));
    let mut base = if base.trim().is_empty() {
        product_key.replace('_', " ")
    } else {
        base
    };
    if had_overlays {
        if let Some(title) = table.title_for(product_key) {
            base = title.to_string();
        } else if let Some((filled, _)) = base.split_once(" / ") {
            base = filled.trim().to_string();
        }
    }
    if !units.is_empty() {
        base = base.replace(&format!(" ({units})"), "");
    }
    let mut title = if units.is_empty() {
        format!("{base} difference, A minus B")
    } else {
        format!("{base} difference ({units}), A minus B")
    };
    if let Some(domain) = domain {
        title.push_str(&format!(" ({domain})"));
    }
    title
}

/// The difference panel's metadata tokens: the shared valid time (and lead
/// when both runs share it), then each run's name with its own
/// initialisation, then run A's remaining tokens.
pub fn difference_subtitle(
    a_left: Option<&str>,
    b_left: Option<&str>,
    labels: &DifferenceLabels,
) -> String {
    let a_tokens = split_tokens(a_left);
    let b_tokens = split_tokens(b_left);
    let find = |tokens: &[String], pick: &dyn Fn(&str) -> bool| {
        tokens.iter().find(|token| pick(token)).cloned()
    };
    let is_init = |token: &str| token.starts_with("Init");
    let is_valid = |token: &str| token.starts_with("Valid ");
    let is_lead = |token: &str| chrome_plan::is_lead_token(token);
    let a_lead = find(&a_tokens, &is_lead);
    let b_lead = find(&b_tokens, &is_lead);
    let same_lead = a_lead == b_lead;
    let mut out: Vec<String> = Vec::new();
    if let Some(valid) = find(&a_tokens, &is_valid) {
        out.push(valid);
    }
    if same_lead {
        if let Some(lead) = a_lead.clone() {
            out.push(lead);
        }
    }
    let run_token = |label: &str, tokens: &[String], lead: &Option<String>| {
        let mut parts = vec![label.to_string()];
        if let Some(init) = find(tokens, &is_init) {
            parts.push(init);
        }
        if !same_lead {
            if let Some(lead) = lead {
                parts.push(lead.clone());
            }
        }
        parts.join(", ")
    };
    out.push(format!("A: {}", run_token(&labels.a, &a_tokens, &a_lead)));
    out.push(format!("B: {}", run_token(&labels.b, &b_tokens, &b_lead)));
    for token in &a_tokens {
        if is_init(token) || is_valid(token) || is_lead(token) {
            continue;
        }
        out.push(token.clone());
    }
    out.join(" | ")
}

/// The difference request: run A's request with its field replaced by
/// A minus B, the zero-centred bar, the difference title and metadata, and
/// no overlays.  Returns the request and the bar's half range, step and rule.
pub fn difference_request(
    a: &MapRenderRequest,
    b: &CapturedPanel,
    product_key: &str,
    labels: &DifferenceLabels,
    table: &DifferenceTable,
) -> Result<(MapRenderRequest, f64, f64, RangeRule), RustwxRenderError> {
    if a.rgba_grid.is_some() || b.rgb {
        return Err(RustwxRenderError::Difference(format!(
            "{product_key} is drawn from an RGB image, which has no value to subtract"
        )));
    }
    if a.legend.mode == LegendMode::Categories || b.categories {
        return Err(RustwxRenderError::Difference(format!(
            "{product_key} draws category codes, and the difference of two codes is not a \
             quantity"
        )));
    }
    let floor = undrawn_floor(&a.scale);
    let values = subtract_fields(&a.field, &b.field, floor)?;
    let units = chrome_plan::display_units(&a.field.units);
    let (half, rule) = half_range(table, product_key, &units, &values);
    let scale = difference_scale(table, half);
    let step = 2.0 * half / table.bands.max(2) as f64;
    let had_overlays = !a.contours.is_empty()
        || !a.wind_barbs.is_empty()
        || !a.wind_streamlines.is_empty();
    let mut request = a.clone();
    let product_name = match &a.field.product {
        ProductKey::Named(name) => name.clone(),
    };
    // Its own product name, so a theme's colormap for the product itself
    // is not laid over the difference bar.
    request.field.product = ProductKey::Named(format!("{product_name}_difference"));
    request.field.values = values;
    request.scale = ColorScale::Discrete(scale);
    request.colorbar = true;
    request.cbar_tick_step = Some(step);
    request.legend = Default::default();
    // Stepped bands, not a smoothed ramp: densifying the palette would
    // blend the neutral bands into their neighbours, and a colour between
    // two bands reads as a difference the bar does not say.
    request.render_density = RenderDensity {
        fill: LevelDensity::default(),
        palette_multiplier: 1,
    };
    request.visual_mode = ProductVisualMode::FilledMeteorology;
    request.rgba_grid = None;
    request.contours.clear();
    request.wind_barbs.clear();
    request.wind_streamlines.clear();
    request.title = Some(difference_title(
        a.title.as_deref(),
        product_key,
        &units,
        had_overlays,
        table,
    ));
    request.subtitle_left = Some(difference_subtitle(
        a.subtitle_left.as_deref(),
        b.subtitle_left.as_deref(),
        labels,
    ));
    Ok((request, half, step, rule))
}

/// The product identity of an output file: its name after the run's own
/// model, cycle and lead (`rustwx_wrf_20250315_12z_f012_d01-3km_2m_temperature`
/// is `d01-3km_2m_temperature`), so one product of two runs with different
/// starts meets under one key.
pub fn difference_key(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let bytes = stem.as_bytes();
    let mut index = 0;
    while let Some(found) = stem[index..].find("z_f") {
        let start = index + found + 3;
        let digits = bytes[start..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        let after = start + digits;
        if digits > 0 && bytes.get(after) == Some(&b'_') {
            return stem[after + 1..].to_string();
        }
        index = start;
    }
    stem
}

/// Where the difference of the product run A would write at `path` goes:
/// the same name with `_difference`, so it files as its own product.
pub fn difference_path(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let extension = path
        .extension()
        .map(|ext| ext.to_string_lossy().into_owned())
        .unwrap_or_else(|| "png".to_string());
    path.with_file_name(format!("{stem}_difference.{extension}"))
}

/// Which half of a difference run is rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferenceStage {
    Capture,
    Subtract,
}

struct Session {
    stage: DifferenceStage,
    labels: DifferenceLabels,
    panel_dir: Option<PathBuf>,
    captured: HashMap<String, CapturedPanel>,
    drawn: Vec<DrawnDifference>,
}

thread_local! {
    static DRAWING_STEPPED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// True while this thread draws a difference panel.  The dense plot styles
/// resample every palette into a smooth ramp, which blends the neutral
/// bands into their neighbours (the band just below zero came out pale
/// blue), so a difference panel keeps its stepped bands under every style.
pub(crate) fn drawing_stepped() -> bool {
    DRAWING_STEPPED.with(std::cell::Cell::get)
}

fn draw_stepped(
    request: &MapRenderRequest,
    path: &Path,
    png_options: &PngWriteOptions,
    plot_style: StaticPlotStyle,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    DRAWING_STEPPED.with(|flag| flag.set(true));
    let result = RustRenderer.save_drawn_png(request, path, png_options, plot_style);
    DRAWING_STEPPED.with(|flag| flag.set(false));
    result
}

fn session() -> &'static Mutex<Option<Session>> {
    static SESSION: OnceLock<Mutex<Option<Session>>> = OnceLock::new();
    SESSION.get_or_init(|| Mutex::new(None))
}

fn lock() -> std::sync::MutexGuard<'static, Option<Session>> {
    session().lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Start run B's pass: every product is kept, not drawn.  With a
/// `panel_dir`, both runs' own pictures are drawn as well, for a
/// three-panel sheet: run B's where its pass writes, run A's in the folder.
pub fn begin_capture(labels: DifferenceLabels, panel_dir: Option<PathBuf>) {
    *lock() = Some(Session {
        stage: DifferenceStage::Capture,
        labels,
        panel_dir,
        captured: HashMap::new(),
        drawn: Vec::new(),
    });
}

/// Start run A's pass over what run B's pass kept.
pub fn begin_subtract() -> Result<usize, RustwxRenderError> {
    let mut guard = lock();
    let session = guard.as_mut().ok_or_else(|| {
        RustwxRenderError::Difference("a difference pass started with no run B captured".into())
    })?;
    session.stage = DifferenceStage::Subtract;
    Ok(session.captured.len())
}

/// End the difference: what was drawn, and the run B products no run A
/// product met.
pub fn finish() -> (Vec<DrawnDifference>, Vec<String>) {
    match lock().take() {
        Some(session) => {
            let mut unmatched: Vec<String> = session.captured.into_keys().collect();
            unmatched.sort();
            (session.drawn, unmatched)
        }
        None => (Vec::new(), Vec::new()),
    }
}

/// The stage in progress, if any.
pub fn stage() -> Option<DifferenceStage> {
    lock().as_ref().map(|session| session.stage)
}

/// A composed multi-panel sheet in a difference run: refused by name.
pub(crate) fn refuse_composed(path: &Path) -> Option<RustwxRenderError> {
    stage().map(|_| {
        RustwxRenderError::Difference(format!(
            "{} is a sheet composed of several panels; a difference is drawn per map product",
            difference_key(path)
        ))
    })
}

/// The save path's hook.  `None` when no difference is in progress.
pub(crate) fn intercept(
    request: &MapRenderRequest,
    path: &Path,
    png_options: &PngWriteOptions,
    plot_style: StaticPlotStyle,
) -> Option<Result<RenderSaveTiming, RustwxRenderError>> {
    let (stage, labels, panel_dir) = {
        let guard = lock();
        let session = guard.as_ref()?;
        (
            session.stage,
            session.labels.clone(),
            session.panel_dir.clone(),
        )
    };
    let key = difference_key(path);
    Some(match stage {
        DifferenceStage::Capture => {
            capture(request, path, &key, panel_dir, png_options, plot_style)
        }
        DifferenceStage::Subtract => subtract(
            request,
            path,
            &key,
            &labels,
            panel_dir,
            png_options,
            plot_style,
        ),
    })
}

fn capture(
    request: &MapRenderRequest,
    path: &Path,
    key: &str,
    panel_dir: Option<PathBuf>,
    png_options: &PngWriteOptions,
    plot_style: StaticPlotStyle,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    // Run B's pass writes into the caller's work folder, never the
    // delivery, and every lane reads back the file it asked for; so a
    // file is always left at `path`: run B's own picture when the run
    // panels were asked for, an empty placeholder otherwise.
    let (timing, panel) = match panel_dir {
        Some(_) => {
            let timing = RustRenderer.save_drawn_png(request, path, png_options, plot_style)?;
            (timing, Some(path.to_path_buf()))
        }
        None => {
            std::fs::write(path, b"").map_err(|source| RustwxRenderError::WriteFile {
                path: path.display().to_string(),
                source,
            })?;
            (
                RenderSaveTiming {
                    georeference_absent_reason: Some(
                        "kept as run B of a difference; nothing was drawn".to_string(),
                    ),
                    ..RenderSaveTiming::default()
                },
                None,
            )
        }
    };
    let captured = CapturedPanel::from_request(request, panel);
    if let Some(session) = lock().as_mut() {
        session.captured.insert(key.to_string(), captured);
    }
    Ok(timing)
}

fn subtract(
    request: &MapRenderRequest,
    path: &Path,
    key: &str,
    labels: &DifferenceLabels,
    panel_dir: Option<PathBuf>,
    png_options: &PngWriteOptions,
    plot_style: StaticPlotStyle,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    let b = lock()
        .as_mut()
        .and_then(|session| session.captured.remove(key))
        .ok_or_else(|| {
            RustwxRenderError::Difference(format!(
                "run B drew no {key} at this valid time, so there is nothing to subtract"
            ))
        })?;
    let table = DifferenceTable::builtin();
    let (diff, half_range, step, rule) = difference_request(request, &b, key, labels, table)?;
    // Written where the lane asked, because every lane reads back the file
    // it asked for; the caller moves it to [`difference_path`] once the
    // product is reported, and `output` names where it will be.
    let timing = draw_stepped(&diff, path, png_options, plot_style)?;
    let output = difference_path(path);
    let a_panel = match panel_dir {
        Some(dir) => {
            let panel = dir.join(format!("{key}_a.png"));
            RustRenderer.save_drawn_png(request, &panel, png_options, plot_style)?;
            Some(panel)
        }
        None => None,
    };
    let defined: Vec<f32> = diff
        .field
        .values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let max_abs = defined
        .iter()
        .fold(0.0f64, |acc, value| acc.max(f64::from(value.abs())));
    let drawn = DrawnDifference {
        key: key.to_string(),
        output,
        a_panel,
        b_panel: b.panel.clone(),
        units: chrome_plan::display_units(&request.field.units),
        half_range,
        step,
        rule,
        defined_cells: defined.len(),
        max_abs,
    };
    if let Some(session) = lock().as_mut() {
        session.drawn.push(drawn);
    }
    Ok(timing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::GridShape;

    fn grid(nx: usize, ny: usize, shift: f32) -> LatLonGrid {
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                lat.push(35.0 + j as f32 * 0.03 + shift);
                lon.push(-100.0 + i as f32 * 0.03);
            }
        }
        LatLonGrid::new(GridShape::new(nx, ny).unwrap(), lat, lon).unwrap()
    }

    fn field(grid: LatLonGrid, units: &str, values: Vec<f32>) -> Field2D {
        Field2D::new(ProductKey::Named("t2".into()), units, grid, values).unwrap()
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|index| 250.0 + (index % 37) as f32 * 0.7).collect()
    }

    #[test]
    fn the_table_parses_and_its_colours_are_one_per_band() {
        let table = DifferenceTable::builtin();
        assert_eq!(table.colors.len(), table.bands);
        assert!(table.neutral_bands < table.bands);
        for hex in &table.colors {
            crate::theme::parse_color(hex).expect("table colour");
        }
        for row in &table.rows {
            assert!(row.half_range > 0.0, "{row:?}");
        }
        assert_eq!(difference_scale(table, 10.0).colors.len(), table.bands);
    }

    #[test]
    fn identical_runs_give_an_all_zero_field_and_a_zero_centred_bar() {
        let values = ramp(12 * 9);
        let a = field(grid(12, 9, 0.0), "K", values.clone());
        let b = field(grid(12, 9, 0.0), "K", values);
        let diff = subtract_fields(&a, &b, None).unwrap();
        assert!(diff.iter().all(|value| *value == 0.0));
        let table = DifferenceTable::builtin();
        let (half, rule) = half_range(table, "d01-3km_unlisted_product", "K", &diff);
        assert_eq!(rule, RangeRule::NoDifference);
        let scale = difference_scale(table, half);
        let levels = &scale.levels;
        assert_eq!(levels.len(), table.bands + 1);
        assert_eq!(levels[table.bands / 2], 0.0, "zero is the middle edge");
        for (low, high) in levels.iter().zip(levels.iter().rev()) {
            assert_eq!(*low, -*high, "the bar is symmetric about zero");
        }
        assert_eq!(scale.extend, ExtendMode::Both);
        // The two bands either side of zero are one neutral colour.
        let mid = table.bands / 2;
        assert_eq!(scale.colors[mid - 1], scale.colors[mid]);
        assert_ne!(scale.colors[mid - 2], scale.colors[mid - 1]);
    }

    #[test]
    fn a_known_offset_gives_that_offset_everywhere() {
        let values = ramp(10 * 8);
        let a = field(grid(10, 8, 0.0), "K", values.iter().map(|v| v + 1.5).collect());
        let b = field(grid(10, 8, 0.0), "K", values);
        let diff = subtract_fields(&a, &b, None).unwrap();
        assert!(diff.iter().all(|value| (value - 1.5).abs() < 1.0e-4), "{diff:?}");
        let reverse = subtract_fields(&b, &a, None).unwrap();
        assert!(reverse.iter().all(|value| (value + 1.5).abs() < 1.0e-4));
        // No table row: the 99th percentile, 1.5, over 5 half-bands is a
        // 0.3 step, rounded up to 0.5, so the bar spans -2.5 to 2.5.
        let (half, rule) = half_range(DifferenceTable::builtin(), "x_unlisted", "K", &diff);
        assert!(matches!(rule, RangeRule::Percentile { .. }), "{rule:?}");
        assert!((half - 2.5).abs() < 1.0e-9, "{half}");
    }

    #[test]
    fn mismatched_grids_are_refused_by_name() {
        let a = field(grid(10, 8, 0.0), "K", ramp(80));
        let b = field(grid(8, 10, 0.0), "K", ramp(80));
        let err = subtract_fields(&a, &b, None).unwrap_err().to_string();
        assert!(err.contains("10x8") && err.contains("8x10"), "{err}");
        assert!(err.contains("never regridded"), "{err}");
        let shifted = field(grid(10, 8, 0.05), "K", ramp(80));
        let err = subtract_fields(&a, &shifted, None).unwrap_err().to_string();
        assert!(err.contains("not one grid"), "{err}");
        let units = field(grid(10, 8, 0.0), "degF", ramp(80));
        let err = subtract_fields(&a, &units, None).unwrap_err().to_string();
        assert!(err.contains("one unit"), "{err}");
    }

    #[test]
    fn values_below_the_floor_difference_as_the_floor_and_empty_in_both_is_undrawn() {
        let g = grid(3, 1, 0.0);
        let a = field(g.clone(), "dBZ", vec![-30.0, 40.0, 20.0]);
        let b = field(g, "dBZ", vec![-20.0, -35.0, 25.0]);
        let diff = subtract_fields(&a, &b, Some(5.0)).unwrap();
        assert!(diff[0].is_nan(), "no echo in either run is undrawn");
        assert_eq!(diff[1], 35.0, "echo against no echo is echo minus the floor");
        assert_eq!(diff[2], -5.0);
    }

    #[test]
    fn a_table_row_fixes_the_range_for_its_product_and_units_only() {
        let table = DifferenceTable::builtin();
        let values = vec![0.3f32; 4];
        let (half, rule) = half_range(table, "d01-3km_2m_temperature", "°F", &values);
        assert_eq!((half, rule), (10.0, RangeRule::Table));
        // Same product in a unit the table does not list: the percentile rule.
        let (_, rule) = half_range(table, "d01-3km_2m_temperature", "furlongs", &values);
        assert!(matches!(rule, RangeRule::Percentile { .. }));
        // A longer product name that merely ends in a listed one's tail.
        assert!(table.row_for("d01-3km_x2m_temperature", "°F").is_none());
    }

    #[test]
    fn the_key_drops_the_run_and_keeps_the_product() {
        let path = Path::new("/o/rustwx_wrf_20250315_12z_f012_d01-3km_2m_temperature.png");
        assert_eq!(difference_key(path), "d01-3km_2m_temperature");
        let other = Path::new("/o/rustwx_wrf_20250316_0z_f000_d01-3km_2m_temperature.png");
        assert_eq!(difference_key(other), difference_key(path));
        assert_eq!(
            difference_path(path),
            Path::new("/o/rustwx_wrf_20250315_12z_f012_d01-3km_2m_temperature_difference.png")
        );
    }

    #[test]
    fn the_title_says_a_minus_b_with_units_and_the_metadata_names_both_runs() {
        let table = DifferenceTable::builtin();
        let title =
            difference_title(Some("2m AGL Temperature (d01 3 km)"), "k", "°F", false, table);
        assert_eq!(title, "2m AGL Temperature difference (°F), A minus B (d01 3 km)");
        // Overlays are not differenced, so the title names the filled field
        // only: by the table's row, or by the part before ` / `.
        let wind = difference_title(
            Some("10m AGL Wind Speed and Direction (d01 3 km)"),
            "d01-3km_10m_wind_speed_and_direction",
            "kt",
            true,
            table,
        );
        assert_eq!(wind, "10m AGL Wind Speed difference (kt), A minus B (d01 3 km)");
        let mslp = difference_title(Some("MSLP / 10m Winds"), "mslp_10m_winds", "hPa", true, table);
        assert_eq!(mslp, "MSLP difference (hPa), A minus B");
        let plain = difference_title(Some("A / B ratio"), "ratio", "", false, table);
        assert_eq!(plain, "A / B ratio difference, A minus B");
        let labels = DifferenceLabels {
            a: "IFS start".into(),
            b: "GFS start".into(),
        };
        let meta = difference_subtitle(
            Some("Init 03/15 12Z | F012 | Valid 03/16 00Z | WRF"),
            Some("Init 03/15 12Z | F012 | Valid 03/16 00Z | WRF"),
            &labels,
        );
        assert_eq!(
            meta,
            "Valid 03/16 00Z | F012 | A: IFS start, Init 03/15 12Z | B: GFS start, Init 03/15 12Z | WRF"
        );
        let staggered = difference_subtitle(
            Some("Init 03/15 12Z | F012 | Valid 03/16 00Z"),
            Some("Init 03/15 18Z | F006 | Valid 03/16 00Z"),
            &labels,
        );
        assert!(staggered.contains("A: IFS start, Init 03/15 12Z, F012"), "{staggered}");
        assert!(staggered.contains("B: GFS start, Init 03/15 18Z, F006"), "{staggered}");
    }
}
