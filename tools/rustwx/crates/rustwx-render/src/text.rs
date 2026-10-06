use crate::color::Rgba;
use font8x8::UnicodeFonts;
use image::RgbaImage;
use rusttype::{Font, Scale, point};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

const SOURCE_SANS_3_REGULAR: &[u8] = include_bytes!("../assets/fonts/SourceSans3-Regular.ttf");
const SOURCE_SANS_3_SEMIBOLD: &[u8] = include_bytes!("../assets/fonts/SourceSans3-Semibold.ttf");
/// The planned chrome's face (Inter 4.0, SIL OFL, `assets/fonts/LICENSE-Inter.txt`).
const INTER_REGULAR: &[u8] = include_bytes!("../assets/fonts/Inter-Regular.ttf");
const INTER_SEMIBOLD: &[u8] = include_bytes!("../assets/fonts/Inter-SemiBold.ttf");

struct FontSet {
    regular: Option<Font<'static>>,
    bold: Option<Font<'static>>,
}

#[derive(Clone, Copy)]
enum FontKind {
    Regular,
    Bold,
}

static FONTS: OnceLock<FontSet> = OnceLock::new();
static CHROME_FONTS: OnceLock<FontSet> = OnceLock::new();

/// The face the planned chrome (header, bar ticks, sheet labels) draws in:
/// a theme's or the environment's font when one is installed, so a themed
/// frame keeps one face throughout, and Inter otherwise.  The fixed-canvas
/// chrome keeps Source Sans 3, so its look does not move.
fn chrome_font(kind: FontKind) -> Option<&'static Font<'static>> {
    let themed = FONT_OVERRIDE
        .get()
        .is_some_and(|installed| installed.regular.is_some() || installed.bold.is_some())
        || env::var_os("RUSTWX_RENDER_FONT_REGULAR").is_some()
        || env::var_os("RUSTWX_RENDER_FONT_BOLD").is_some();
    if themed {
        return get_font(kind);
    }
    let fonts = CHROME_FONTS.get_or_init(|| FontSet {
        regular: Font::try_from_bytes(INTER_REGULAR),
        bold: Font::try_from_bytes(INTER_SEMIBOLD),
    });
    match kind {
        FontKind::Regular => fonts.regular.as_ref(),
        FontKind::Bold => fonts.bold.as_ref().or(fonts.regular.as_ref()),
    }
    .or_else(|| get_font(kind))
}

/// Font files a theme installed before the first glyph was drawn.  Read
/// ahead of the environment override and the embedded faces; a path that
/// does not load is reported once and that weight keeps the embedded face.
static FONT_OVERRIDE: OnceLock<FontOverride> = OnceLock::new();

#[derive(Debug, Clone, Default)]
struct FontOverride {
    regular: Option<PathBuf>,
    bold: Option<PathBuf>,
}

/// Install theme font paths.  A no-op after the fonts have been loaded
/// (the first text draw), which is why callers install themes first.
pub fn install_font_override(regular: Option<PathBuf>, bold: Option<PathBuf>) {
    let _ = FONT_OVERRIDE.set(FontOverride { regular, bold });
}

pub fn draw_text(img: &mut RgbaImage, text: &str, x: i32, y: i32, color: Rgba, scale: u32) {
    draw_text_inner(img, text, x, y, color, scale, 1.0, FontKind::Regular);
}

pub fn draw_text_bold(img: &mut RgbaImage, text: &str, x: i32, y: i32, color: Rgba, scale: u32) {
    draw_text_inner(img, text, x, y, color, scale, 1.0, FontKind::Bold);
}

pub fn draw_text_centered(img: &mut RgbaImage, text: &str, y: i32, color: Rgba, scale: u32) {
    let w = text_width_bold(text, scale);
    let x = ((img.width() as i32) - w as i32) / 2;
    draw_text_bold(img, text, x, y, color, scale);
}

pub fn draw_text_right(
    img: &mut RgbaImage,
    text: &str,
    x_right: i32,
    y: i32,
    color: Rgba,
    scale: u32,
) {
    let w = text_width(text, scale);
    draw_text_inner(
        img,
        text,
        x_right - w as i32,
        y,
        color,
        scale,
        1.0,
        FontKind::Regular,
    );
}

pub fn text_width(text: &str, scale: u32) -> u32 {
    measure_text(text, scale, 1.0, FontKind::Regular)
}

pub(crate) fn draw_text_right_with_factor(
    img: &mut RgbaImage,
    text: &str,
    x_right: i32,
    y: i32,
    color: Rgba,
    scale: u32,
    size_factor: f32,
) {
    let w = measure_text(text, scale, size_factor, FontKind::Regular);
    draw_text_inner(
        img,
        text,
        x_right - w as i32,
        y,
        color,
        scale,
        size_factor,
        FontKind::Regular,
    );
}

pub fn text_width_bold(text: &str, scale: u32) -> u32 {
    measure_text(text, scale, 1.0, FontKind::Bold)
}

pub(crate) fn regular_line_height(scale: u32) -> u32 {
    line_height(scale, 1.0, FontKind::Regular)
}

pub(crate) fn bold_line_height(scale: u32) -> u32 {
    line_height(scale, 1.0, FontKind::Bold)
}

pub(crate) fn draw_text_with_factor(
    img: &mut RgbaImage,
    text: &str,
    x: i32,
    y: i32,
    color: Rgba,
    scale: u32,
    size_factor: f32,
) {
    draw_text_inner(
        img,
        text,
        x,
        y,
        color,
        scale,
        size_factor,
        FontKind::Regular,
    );
}

pub(crate) fn draw_text_bold_with_factor(
    img: &mut RgbaImage,
    text: &str,
    x: i32,
    y: i32,
    color: Rgba,
    scale: u32,
    size_factor: f32,
) {
    draw_text_inner(img, text, x, y, color, scale, size_factor, FontKind::Bold);
}

pub(crate) fn text_width_with_factor(text: &str, scale: u32, size_factor: f32) -> u32 {
    measure_text(text, scale, size_factor, FontKind::Regular)
}

pub(crate) fn text_width_bold_with_factor(text: &str, scale: u32, size_factor: f32) -> u32 {
    measure_text(text, scale, size_factor, FontKind::Bold)
}

pub(crate) fn regular_line_height_with_factor(scale: u32, size_factor: f32) -> u32 {
    line_height(scale, size_factor, FontKind::Regular)
}

pub(crate) struct VerticalText {
    pub width: u32,
    pub height: u32,
    pixels: Vec<(u32, u32, u8)>,
}

impl VerticalText {
    pub fn draw(&self, img: &mut RgbaImage, x: u32, y: u32, color: Rgba) {
        for &(dx, dy, coverage) in &self.pixels {
            let mut ink = color;
            ink.a = ((u32::from(color.a) * u32::from(coverage) + 127) / 255) as u8;
            blend_pixel(img, (x + dx) as i32, (y + dy) as i32, ink);
        }
    }
}

pub(crate) fn vertical_text(text: &str, scale: u32, size_factor: f32) -> VerticalText {
    // Rasterize through the same font owner, then rotate coverage rather
    // than a canvas rectangle. Padding retains glyph overhangs; trimming
    // only zero-coverage pixels gives the actual complete label bounds.
    let pad = regular_line_height_with_factor(scale, size_factor).max(1);
    let width = text_width_with_factor(text, scale, size_factor) + 2 * pad;
    let height = 3 * pad;
    let mut mask = RgbaImage::from_pixel(width, height, Rgba::BLACK.to_image_rgba());
    draw_text_with_factor(&mut mask, text, pad as i32, pad as i32,
                          Rgba::WHITE, scale, size_factor);
    let (mut left, mut top, mut right, mut bottom) = (width, height, 0, 0);
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel[0] > 0 {
            left = left.min(x); top = top.min(y);
            right = right.max(x); bottom = bottom.max(y);
        }
    }
    if left > right || top > bottom {
        return VerticalText { width: 0, height: 0, pixels: Vec::new() };
    }
    let pixels = mask.enumerate_pixels().filter_map(|(x, y, pixel)| {
        if pixel[0] > 0 { Some((y - top, right - x, pixel[0])) } else { None }
    }).collect();
    VerticalText { width: bottom - top + 1, height: right - left + 1, pixels }
}

pub(crate) fn bold_line_height_with_factor(scale: u32, size_factor: f32) -> u32 {
    line_height(scale, size_factor, FontKind::Bold)
}

/// One value's label: whole numbers bare, anything else to one decimal.
///
/// A colour bar labels its ticks as a SET through [`format_tick_labels`],
/// which starts from these labels and adds places only where they stop
/// reading as their ticks. This alone is right for a lone number (a
/// contour label, an extreme marker) whose neighbours are not printed
/// beside it.
pub fn format_tick(value: f64) -> String {
    if value == value.floor() {
        format!("{}", value as i64)
    } else {
        fixed_places_label(value, 1)
    }
}

/// `value` to `places` decimals, trailing zeros and a bare point dropped,
/// and a value that rounds to zero written `0`, never `-0`.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): a colour bar crossing zero
/// on a fractional step prints `-0` at its zero tick. The ticks are
/// stepped by repeated addition, so the one meant to be zero arrives as
/// about -3e-17, and the old trim of `-0.0` left `-0`.
fn fixed_places_label(value: f64, places: usize) -> String {
    let fixed = format!("{value:.places$}");
    let trimmed = if fixed.contains('.') {
        fixed.trim_end_matches('0').trim_end_matches('.')
    } else {
        fixed.as_str()
    };
    if trimmed == "-0" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// The places past which a tick label stops being a readable number. A
/// set whose spacing needs more than this falls back to each value's
/// shortest exact spelling.
const MAX_TICK_PLACES: usize = 12;

/// The labels for one colour bar's ticks, formatted together so every
/// label reads as its own tick.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): a colour bar whose distinct
/// ticks print the same number. [`format_tick`] carries one decimal, so
/// a narrow range around a large value -- 200 hPa height drawn from
/// 12.200 to 12.228 against `1e3 gpm` -- printed `12.2` at all fourteen
/// ticks, and mean sea level pressure, the freezing level and the
/// 250 to 850 hPa heights did the same on ordinary forecasts. A bar on
/// quarter steps printed 0.25 as `0.2` and 0.75 as `0.8`: different
/// labels, but not the ticks' values.
///
/// The rule is about what a label says, not about the field: every label
/// must parse back to its tick to within a hundredth of the spacing
/// between neighbouring ticks. A bar whose usual labels already do that
/// keeps them unchanged, so an ordinary bar draws exactly as before; a
/// bar whose labels do not gets the fewest extra places that make every
/// one of them true.
pub fn format_tick_labels(ticks: &[f64]) -> Vec<String> {
    let usual: Vec<String> = ticks.iter().map(|value| format_tick(*value)).collect();
    let Some(spacing) = smallest_tick_spacing(ticks) else {
        return usual;
    };
    let tolerance = spacing * 0.01;
    let reads_as_ticks = |labels: &[String]| {
        labels.iter().zip(ticks).all(|(label, value)| {
            !value.is_finite()
                || label
                    .parse::<f64>()
                    .is_ok_and(|shown| (shown - value).abs() <= tolerance)
        })
    };
    if reads_as_ticks(&usual) {
        return usual;
    }
    for places in 2..=MAX_TICK_PLACES {
        let labels: Vec<String> = ticks
            .iter()
            .map(|value| {
                if value.is_finite() {
                    fixed_places_label(*value, places)
                } else {
                    format_tick(*value)
                }
            })
            .collect();
        if reads_as_ticks(&labels) {
            return labels;
        }
    }
    ticks
        .iter()
        .map(|value| {
            if value.is_finite() {
                let exact = value.to_string();
                if exact == "-0" { "0".to_string() } else { exact }
            } else {
                format_tick(*value)
            }
        })
        .collect()
}

/// The smallest gap between two different finite ticks, or `None` when
/// the set holds fewer than two different values (nothing to tell apart).
fn smallest_tick_spacing(ticks: &[f64]) -> Option<f64> {
    let mut finite: Vec<f64> = ticks.iter().copied().filter(|value| value.is_finite()).collect();
    finite.sort_by(f64::total_cmp);
    finite
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|gap| *gap > 0.0)
        .min_by(f64::total_cmp)
}

fn draw_text_inner(
    img: &mut RgbaImage,
    text: &str,
    x: i32,
    y: i32,
    color: Rgba,
    scale: u32,
    size_factor: f32,
    kind: FontKind,
) {
    if let Some(font) = get_font(kind) {
        draw_ttf_text(img, text, x, y, color, scale, size_factor, font, kind);
    } else {
        draw_bitmap_text(
            img,
            text,
            x,
            y,
            color,
            effective_bitmap_scale(scale, size_factor),
        );
    }
}

fn measure_text(text: &str, scale: u32, size_factor: f32, kind: FontKind) -> u32 {
    if let Some(font) = get_font(kind) {
        let scale = Scale::uniform(font_size_px(scale, size_factor, kind));
        let v_metrics = font.v_metrics(scale);
        let glyphs: Vec<_> = font
            .layout(text, scale, point(0.0, v_metrics.ascent))
            .collect();
        glyphs
            .iter()
            .rev()
            .find_map(|g| g.pixel_bounding_box().map(|bb| bb.max.x.max(0) as u32))
            .or_else(|| {
                glyphs.last().map(|g| {
                    let end = g.position().x + g.unpositioned().h_metrics().advance_width;
                    end.max(0.0).ceil() as u32
                })
            })
            .unwrap_or(0)
    } else {
        text.len() as u32 * 8 * effective_bitmap_scale(scale, size_factor)
    }
}

fn draw_ttf_text(
    img: &mut RgbaImage,
    text: &str,
    x: i32,
    y: i32,
    color: Rgba,
    scale_tag: u32,
    size_factor: f32,
    font: &Font<'static>,
    kind: FontKind,
) {
    let scale = match font_size_px(scale_tag, size_factor, kind) {
        s if s > 0.0 => Scale::uniform(s),
        _ => Scale::uniform(12.0),
    };
    let v_metrics = font.v_metrics(scale);
    let glyphs = font.layout(text, scale, point(x as f32, y as f32 + v_metrics.ascent));

    for glyph in glyphs {
        if let Some(bb) = glyph.pixel_bounding_box() {
            glyph.draw(|gx, gy, coverage| {
                let px = bb.min.x + gx as i32;
                let py = bb.min.y + gy as i32;
                let alpha = ((color.a as f32) * coverage).round().clamp(0.0, 255.0) as u8;
                blend_pixel(
                    img,
                    px,
                    py,
                    Rgba {
                        r: color.r,
                        g: color.g,
                        b: color.b,
                        a: alpha,
                    },
                );
            });
        }
    }
}

fn draw_bitmap_text(img: &mut RgbaImage, text: &str, x: i32, y: i32, color: Rgba, scale: u32) {
    let ic = color.to_image_rgba();
    let char_w = 8 * scale;

    for (ci, ch) in text.chars().enumerate() {
        let glyph = get_bitmap_glyph(ch);
        let cx = x + (ci as i32) * char_w as i32;

        for row in 0..8u32 {
            let bits = glyph[row as usize];
            for col in 0..8u32 {
                if bits & (1 << col) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cx + (col * scale + sx) as i32;
                            let py = y + (row * scale + sy) as i32;
                            if px >= 0
                                && py >= 0
                                && (px as u32) < img.width()
                                && (py as u32) < img.height()
                            {
                                img.put_pixel(px as u32, py as u32, ic);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn get_bitmap_glyph(ch: char) -> [u8; 8] {
    if (ch as u32) < 128 {
        font8x8::BASIC_FONTS.get(ch).unwrap_or([0u8; 8])
    } else {
        [0u8; 8]
    }
}

fn blend_pixel(img: &mut RgbaImage, x: i32, y: i32, color: Rgba) {
    if x < 0 || y < 0 || (x as u32) >= img.width() || (y as u32) >= img.height() {
        return;
    }
    if color.a == 0 {
        return;
    }

    let dst = img.get_pixel(x as u32, y as u32).0;
    let alpha = color.a as f64 / 255.0;
    let inv = 1.0 - alpha;
    let blended = image::Rgba([
        (color.r as f64 * alpha + dst[0] as f64 * inv).round() as u8,
        (color.g as f64 * alpha + dst[1] as f64 * inv).round() as u8,
        (color.b as f64 * alpha + dst[2] as f64 * inv).round() as u8,
        255,
    ]);
    img.put_pixel(x as u32, y as u32, blended);
}

fn font_size_px(scale: u32, size_factor: f32, kind: FontKind) -> f32 {
    let base = match (scale.max(1), kind) {
        (1, FontKind::Regular) => 12.0,
        (1, FontKind::Bold) => 15.0,
        (2, FontKind::Regular) => 16.0,
        (2, FontKind::Bold) => 19.0,
        (s, FontKind::Regular) => 12.0 + (s as f32 - 1.0) * 4.0,
        (s, FontKind::Bold) => 15.0 + (s as f32 - 1.0) * 4.0,
    };
    base * size_factor.clamp(0.65, 2.0)
}

fn effective_bitmap_scale(scale: u32, size_factor: f32) -> u32 {
    ((scale.max(1) as f32) * size_factor.clamp(0.65, 2.0))
        .round()
        .max(1.0) as u32
}

fn get_font(kind: FontKind) -> Option<&'static Font<'static>> {
    let fonts = FONTS.get_or_init(load_fonts);
    match kind {
        FontKind::Regular => fonts.regular.as_ref(),
        FontKind::Bold => fonts.bold.as_ref().or(fonts.regular.as_ref()),
    }
}

fn load_fonts() -> FontSet {
    FontSet {
        regular: load_font(false),
        bold: load_font(true),
    }
}

fn load_font(bold: bool) -> Option<Font<'static>> {
    load_font_override(bold)
        .or_else(|| load_embedded_font(bold))
        .or_else(|| load_font_candidates(bold))
}

fn load_font_override(bold: bool) -> Option<Font<'static>> {
    if let Some(installed) = FONT_OVERRIDE.get() {
        let path = if bold {
            installed.bold.as_ref()
        } else {
            installed.regular.as_ref()
        };
        if let Some(path) = path {
            match load_font_from_path(path.clone()) {
                Some(font) => return Some(font),
                None => eprintln!(
                    "THEME_FONT\t{} font {} did not load; using the embedded face",
                    if bold { "bold" } else { "regular" },
                    path.display()
                ),
            }
        }
    }
    let env_keys = if bold {
        ["RUSTWX_RENDER_FONT_BOLD", "WRF_RENDER_FONT_BOLD"]
    } else {
        ["RUSTWX_RENDER_FONT_REGULAR", "WRF_RENDER_FONT_REGULAR"]
    };
    env_keys
        .iter()
        .find_map(|key| env::var(key).ok())
        .and_then(|value| load_font_from_path(PathBuf::from(value)))
}

fn load_embedded_font(bold: bool) -> Option<Font<'static>> {
    let bytes = if bold {
        SOURCE_SANS_3_SEMIBOLD
    } else {
        SOURCE_SANS_3_REGULAR
    };
    Font::try_from_bytes(bytes)
}

fn load_font_candidates(bold: bool) -> Option<Font<'static>> {
    for path in font_candidates(bold) {
        if let Some(font) = load_font_from_path(path) {
            return Some(font);
        }
    }
    None
}

fn load_font_from_path(path: PathBuf) -> Option<Font<'static>> {
    fs::read(path).ok().and_then(Font::try_from_vec)
}

fn font_candidates(bold: bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let dejavu_name = if bold {
        "DejaVuSans-Bold.ttf"
    } else {
        "DejaVuSans.ttf"
    };
    let liberation_name = if bold {
        "LiberationSans-Bold.ttf"
    } else {
        "LiberationSans-Regular.ttf"
    };
    let noto_name = if bold {
        "NotoSans-Bold.ttf"
    } else {
        "NotoSans-Regular.ttf"
    };
    let arial_name = if bold { "arialbd.ttf" } else { "arial.ttf" };
    let segoe_name = if bold { "segoeuib.ttf" } else { "segoeui.ttf" };

    if let Ok(xdg_data_home) = env::var("XDG_DATA_HOME") {
        out.push(
            PathBuf::from(&xdg_data_home)
                .join("fonts")
                .join(dejavu_name),
        );
        out.push(
            PathBuf::from(&xdg_data_home)
                .join("fonts")
                .join(liberation_name),
        );
        out.push(PathBuf::from(&xdg_data_home).join("fonts").join(noto_name));
    }

    if let Ok(home) = env::var("HOME") {
        let home = PathBuf::from(home);
        out.push(
            home.join(".local")
                .join("share")
                .join("fonts")
                .join(dejavu_name),
        );
        out.push(
            home.join(".local")
                .join("share")
                .join("fonts")
                .join(liberation_name),
        );
        out.push(
            home.join(".local")
                .join("share")
                .join("fonts")
                .join(noto_name),
        );
        out.push(home.join(".fonts").join(dejavu_name));
        out.push(home.join(".fonts").join(liberation_name));
        out.push(home.join(".fonts").join(noto_name));
    }

    if let Ok(home) = env::var("USERPROFILE") {
        let home = PathBuf::from(home);
        let mpl = home
            .join("AppData")
            .join("Roaming")
            .join("Python")
            .join("Python313")
            .join("site-packages")
            .join("matplotlib")
            .join("mpl-data")
            .join("fonts")
            .join("ttf");
        out.push(mpl.join(dejavu_name));
        out.push(
            home.join("AppData")
                .join("Local")
                .join("Microsoft")
                .join("Windows")
                .join("Fonts")
                .join(dejavu_name),
        );
    }

    out.push(PathBuf::from("/usr/share/fonts/truetype/dejavu").join(dejavu_name));
    out.push(PathBuf::from("/usr/share/fonts/dejavu").join(dejavu_name));
    out.push(PathBuf::from("/usr/share/fonts/truetype/liberation2").join(liberation_name));
    out.push(PathBuf::from("/usr/share/fonts/truetype/liberation").join(liberation_name));
    out.push(PathBuf::from("/usr/share/fonts/truetype/noto").join(noto_name));
    out.push(PathBuf::from("/usr/share/fonts/opentype/noto").join(noto_name));
    out.push(PathBuf::from("/usr/local/share/fonts").join(dejavu_name));
    out.push(PathBuf::from("/usr/local/share/fonts").join(liberation_name));
    out.push(PathBuf::from("/mnt/c/Windows/Fonts").join(arial_name));
    out.push(PathBuf::from("/mnt/c/Windows/Fonts").join(segoe_name));
    out.push(
        PathBuf::from(r"C:\Python313\Lib\site-packages\matplotlib\mpl-data\fonts\ttf")
            .join(dejavu_name),
    );
    out.push(PathBuf::from(r"C:\Windows\Fonts").join(segoe_name));
    out.push(PathBuf::from(r"C:\Windows\Fonts").join(arial_name));

    out
}

/// Chrome text at an exact pixel size, the size the layout table names.
///
/// Digits are set in cells of one width (the widest digit), so a lead
/// time or a valid time that changes from frame to frame does not move
/// the text after it: `F009` to `F010` stays put in a loop.  Every other
/// glyph keeps its own advance and kerning.
pub(crate) fn draw_text_px(
    img: &mut RgbaImage,
    text: &str,
    x: i32,
    y: i32,
    color: Rgba,
    size_px: f32,
    bold: bool,
) {
    let kind = if bold { FontKind::Bold } else { FontKind::Regular };
    let Some(font) = chrome_font(kind) else {
        draw_bitmap_text(img, text, x, y, color, ((size_px / 12.0).round() as u32).max(1));
        return;
    };
    let scale = Scale::uniform(size_px.max(1.0));
    let ascent = font.v_metrics(scale).ascent;
    for (glyph, _) in tabular_layout(font, text, scale, x as f32, y as f32 + ascent) {
        if let Some(bb) = glyph.pixel_bounding_box() {
            glyph.draw(|gx, gy, coverage| {
                let alpha = ((color.a as f32) * coverage).round().clamp(0.0, 255.0) as u8;
                blend_pixel(
                    img,
                    bb.min.x + gx as i32,
                    bb.min.y + gy as i32,
                    Rgba { a: alpha, ..color },
                );
            });
        }
    }
}

/// Width `draw_text_px` gives `text`: the advance of the last glyph,
/// so right alignment lands on the same pixel frame after frame.
pub(crate) fn text_width_px(text: &str, size_px: f32, bold: bool) -> u32 {
    let kind = if bold { FontKind::Bold } else { FontKind::Regular };
    let Some(font) = chrome_font(kind) else {
        return text.chars().count() as u32 * 8 * ((size_px / 12.0).round() as u32).max(1);
    };
    let scale = Scale::uniform(size_px.max(1.0));
    tabular_layout(font, text, scale, 0.0, 0.0)
        .last()
        .map(|(glyph, advance)| (glyph.position().x + advance).ceil().max(0.0) as u32)
        .unwrap_or(0)
}

pub(crate) fn ascent_px(size_px: f32, bold: bool) -> f32 {
    let kind = if bold { FontKind::Bold } else { FontKind::Regular };
    match chrome_font(kind) {
        Some(font) => font.v_metrics(Scale::uniform(size_px.max(1.0))).ascent,
        None => size_px * 0.8,
    }
}

pub(crate) fn line_height_px(size_px: f32, bold: bool) -> u32 {
    let kind = if bold { FontKind::Bold } else { FontKind::Regular };
    match chrome_font(kind) {
        Some(font) => {
            let metrics = font.v_metrics(Scale::uniform(size_px.max(1.0)));
            (metrics.ascent - metrics.descent).ceil().max(size_px.ceil()) as u32
        }
        None => (size_px.ceil() as u32).max(8),
    }
}

/// Whether the embedded chrome faces draw every glyph of `text` (no
/// glyph falls back to the empty `.notdef` box).
pub fn chrome_font_covers(text: &str) -> bool {
    [FontKind::Regular, FontKind::Bold].iter().all(|kind| match chrome_font(*kind) {
        Some(font) => text
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .all(|ch| font.glyph(ch).id().0 != 0),
        None => text.is_ascii(),
    })
}

fn tabular_layout<'a>(
    font: &'a Font<'static>,
    text: &str,
    scale: Scale,
    x: f32,
    baseline: f32,
) -> Vec<(rusttype::PositionedGlyph<'a>, f32)> {
    let digit_advance = ('0'..='9')
        .map(|digit| font.glyph(digit).scaled(scale).h_metrics().advance_width)
        .fold(0.0f32, f32::max);
    let mut out = Vec::with_capacity(text.len());
    let mut caret = x;
    let mut previous: Option<rusttype::GlyphId> = None;
    for ch in text.chars() {
        let glyph = font.glyph(ch).scaled(scale);
        let id = glyph.id();
        let natural = glyph.h_metrics().advance_width;
        let digit = ch.is_ascii_digit();
        if let (Some(prev), false) = (previous, digit) {
            caret += font.pair_kerning(scale, prev, id);
        }
        let (offset, advance) = if digit {
            ((digit_advance - natural) / 2.0, digit_advance)
        } else {
            (0.0, natural)
        };
        out.push((glyph.positioned(point(caret + offset, baseline)), advance - offset));
        caret += advance;
        previous = Some(id);
    }
    out
}

fn line_height(scale: u32, size_factor: f32, kind: FontKind) -> u32 {
    if let Some(font) = get_font(kind) {
        let px = font_size_px(scale, size_factor, kind);
        let scale = Scale::uniform(px);
        let metrics = font.v_metrics(scale);
        (metrics.ascent - metrics.descent + metrics.line_gap)
            .ceil()
            .max(px.ceil()) as u32
    } else {
        (8 * effective_bitmap_scale(scale, size_factor)).max(12)
    }
}

#[cfg(test)]
mod tests {
    use super::{FontKind, get_font, line_height, load_embedded_font, text_width};

    #[test]
    fn embedded_source_sans_fonts_load() {
        assert!(load_embedded_font(false).is_some());
        assert!(load_embedded_font(true).is_some());
    }

    #[test]
    fn renderer_has_outline_fonts_available_by_default() {
        assert!(get_font(FontKind::Regular).is_some());
        assert!(get_font(FontKind::Bold).is_some());
        assert!(text_width("RustWX", 1) > 0);
        assert!(line_height(1, 1.0, FontKind::Regular) >= 12);
    }
}
