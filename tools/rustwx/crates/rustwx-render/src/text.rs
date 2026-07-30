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

pub(crate) fn bold_line_height_with_factor(scale: u32, size_factor: f32) -> u32 {
    line_height(scale, size_factor, FontKind::Bold)
}

pub fn format_tick(value: f64) -> String {
    if value == value.floor() {
        format!("{}", value as i64)
    } else {
        let s = format!("{:.1}", value);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
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
