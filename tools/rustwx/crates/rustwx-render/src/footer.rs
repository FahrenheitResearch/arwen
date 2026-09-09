//! The footer strip: a band drawn UNDER a finished panel carrying a logo and
//! two caption lines.
//!
//! WHAT BREAKAGE THIS PREVENTS (gate law): a delivered sheet that
//! does not say what it is.  The panel's own title row is sized and placed by
//! the plot layout and carries the product and the valid time in the
//! renderer's voice; a gallery handed to a reader also has to carry the
//! producer's mark, the mesh or grid the numbers came from, which leg of a
//! pair is drawn and the band the case is about.  Writing those into the
//! title row would move every existing pixel; the strip is appended below the
//! canvas instead, so `map_x`/`map_y` and therefore the published
//! georeference stay exactly where the render put them.
//!
//! It is drawn once, in the save path every product family already goes
//! through, and only when BOTH halves are present: a theme with a `footer`
//! section (neither built-in has one) and caption fields installed by the
//! caller.  Absent either, nothing is composed and the encoded bytes are the
//! bytes the renderer produced -- which is what keeps the pixel-regression
//! gate green.

use crate::color::Rgba;
use crate::text;
use crate::theme::{FOOTER_FIELDS, FooterTheme};
use image::RgbaImage;
use image::imageops::{FilterType, resize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};

/// The caption fields a footer template may name.  Every field is optional:
/// a leg that has no band note leaves `note` unset and the segment holding
/// it is dropped along with its separator.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FooterFields {
    pub product_title: Option<String>,
    pub valid_time: Option<String>,
    pub mesh_or_grid: Option<String>,
    pub leg: Option<String>,
    pub note: Option<String>,
}

impl FooterFields {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    fn get(&self, name: &str) -> Option<&str> {
        let value = match name {
            "product_title" => self.product_title.as_deref(),
            "valid_time" => self.valid_time.as_deref(),
            "mesh_or_grid" => self.mesh_or_grid.as_deref(),
            "leg" => self.leg.as_deref(),
            "note" => self.note.as_deref(),
            _ => None,
        };
        value.map(str::trim).filter(|text| !text.is_empty())
    }

    /// Fill the fields the caller left unset from the panel's own headline,
    /// so a caller that installs nothing but a leg still gets a strip that
    /// names the product and the time.
    ///
    /// The left subtitle is the renderer's own row, `Init A | F0NN | Valid B
    /// | MODEL | dx C`.  Only the valid segment belongs in a caption whose
    /// next field is the grid: taking the whole row put the init, the lead,
    /// the model and the spacing in the time slot and then repeated the
    /// spacing two fields later.
    pub fn with_panel_defaults(mut self, title: Option<&str>, subtitle: Option<&str>) -> Self {
        if self.product_title.is_none() {
            self.product_title = title.map(str::to_string);
        }
        if self.valid_time.is_none() {
            self.valid_time = subtitle.map(valid_segment);
        }
        self
    }
}

/// The `Valid ...` segment of a subtitle row, or the row itself when it
/// carries no such segment.
fn valid_segment(subtitle: &str) -> String {
    subtitle
        .split('|')
        .map(str::trim)
        .find(|segment| {
            segment
                .split_whitespace()
                .next()
                .map(|word| word.eq_ignore_ascii_case("valid"))
                .unwrap_or(false)
        })
        .map(|segment| segment[5..].trim().to_string())
        .unwrap_or_else(|| subtitle.trim().to_string())
}

static FOOTER_FIELDS_STATE: RwLock<Option<FooterFields>> = RwLock::new(None);

/// Install the caption fields for the renders that follow.  Called once per
/// frame by a batch lane, because the valid time and the leg change with the
/// frame while the theme does not.
pub fn set_footer_fields(fields: FooterFields) {
    if let Ok(mut slot) = FOOTER_FIELDS_STATE.write() {
        *slot = Some(fields);
    }
}

/// Clear the installed caption fields; the next render draws no strip.
pub fn clear_footer_fields() {
    if let Ok(mut slot) = FOOTER_FIELDS_STATE.write() {
        *slot = None;
    }
}

/// What [`set_footer_fields`] installed, if anything.
pub fn footer_fields() -> Option<FooterFields> {
    FOOTER_FIELDS_STATE.read().ok().and_then(|slot| slot.clone())
}

/// Substitute `{field}` names, then drop every `|`-separated segment whose
/// text is empty once the fields are filled.
///
/// The collapse is what lets ONE template serve a control leg, a treatment
/// leg and a difference leg: the template names every field it could carry
/// and the segments that stayed blank do not leave a trail of separators on
/// the sheet.
pub fn render_template(template: &str, fields: &FooterFields) -> String {
    // The TEMPLATE is split, then each segment is substituted -- never the
    // other way round.  A field's own value may carry a separator (the
    // renderer's subtitle row is itself `A | B | C`), and splitting after
    // substitution turned one field into three segments and dropped the
    // wrong ones.
    let segments: Vec<&str> = template.split('|').map(str::trim).collect();
    let mut kept: Vec<String> = Vec::with_capacity(segments.len());
    for segment in segments {
        let named = field_names(segment);
        if !named.is_empty() && named.iter().all(|name| fields.get(name).is_none()) {
            continue;
        }
        let filled = substitute(segment, fields);
        let filled = filled.trim();
        if !filled.is_empty() {
            kept.push(filled.to_string());
        }
    }
    kept.join(" | ")
}

/// The footer fields one template segment names.
fn field_names(segment: &str) -> Vec<String> {
    let chars: Vec<char> = segment.chars().collect();
    let mut names = Vec::new();
    let mut index = 0usize;
    while index < chars.len() {
        if chars[index] != '{' {
            index += 1;
            continue;
        }
        let start = index + 1;
        let mut end = start;
        while end < chars.len() && chars[end] != '}' {
            end += 1;
        }
        if end >= chars.len() {
            break;
        }
        let name: String = chars[start..end].iter().collect();
        if FOOTER_FIELDS.contains(&name.as_str()) {
            names.push(name);
        }
        index = end + 1;
    }
    names
}

fn substitute(template: &str, fields: &FooterFields) -> String {
    let mut out = String::with_capacity(template.len());
    let chars: Vec<char> = template.chars().collect();
    let mut index = 0usize;
    while index < chars.len() {
        if chars[index] != '{' {
            out.push(chars[index]);
            index += 1;
            continue;
        }
        let start = index + 1;
        let mut end = start;
        while end < chars.len() && chars[end] != '}' {
            end += 1;
        }
        if end >= chars.len() {
            // The theme resolver refuses an unclosed brace, so this arm is
            // only reachable through a hand-built FooterTheme; copy the text
            // rather than eat the rest of the line.
            out.extend(&chars[index..]);
            break;
        }
        let name: String = chars[start..end].iter().collect();
        if FOOTER_FIELDS.contains(&name.as_str()) {
            if let Some(value) = fields.get(&name) {
                out.push_str(value);
            }
        } else {
            out.push('{');
            out.push_str(&name);
            out.push('}');
        }
        index = end + 1;
    }
    out
}

/// A decoded logo, cached per path for the life of the process: a batch lane
/// draws one strip per frame and re-decoding the PNG per frame is work with
/// no answer attached to it.
fn logo_for(path: &Path) -> Option<&'static RgbaImage> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Option<&'static RgbaImage>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().ok()?;
    if let Some(entry) = cache.get(path) {
        return *entry;
    }
    let decoded = decode_logo(path);
    if decoded.is_none() {
        eprintln!(
            "FOOTER_LOGO_ABSENT\t{}\tthe strip draws its caption without the mark",
            path.display()
        );
    }
    let leaked: Option<&'static RgbaImage> = decoded.map(|image| &*Box::leak(Box::new(image)));
    cache.insert(path.to_path_buf(), leaked);
    leaked
}

fn decode_logo(path: &Path) -> Option<RgbaImage> {
    let bytes = std::fs::read(path).ok()?;
    let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).ok()?;
    Some(decoded.to_rgba8())
}

/// The strip's own text sizes, derived from its height so a taller band is
/// not a band of the same small type with more air around it.
fn text_factors(height: u32) -> (f32, f32) {
    let title_px = (height as f32 * 0.26).clamp(13.0, 64.0);
    let caption_px = (height as f32 * 0.19).clamp(11.0, 48.0);
    // font_size_px: bold scale 1 is 15 px, regular scale 1 is 12 px.
    (
        (title_px / 15.0).clamp(0.85, 4.5),
        (caption_px / 12.0).clamp(0.9, 4.0),
    )
}

/// A strip at least this many pixels tall for every 1800 pixels of image
/// width: the band the reviewer rejected was 64 px under an 1800 px sheet.
const STRIP_MIN_PX_AT_1800: f32 = 110.0;

/// The strip is at least this fraction of the image's own height.
const STRIP_MIN_HEIGHT_FRACTION: f32 = 0.07;

/// The wordmark takes about this much of the strip's height.
const WORDMARK_HEIGHT_FRACTION: f32 = 0.70;

/// The strip's height and its wordmark's height for an image of this size.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): the theme named ONE
/// pixel height and every sheet got it, so a 64-pixel band with small type
/// hung under an 1800-pixel delivery and read, in the reviewer's words, as a bar so
/// tiny you cannot see anything.  The geometry is the image's now; the
/// theme's own `height` and `logo_height` survive as FLOORS, so a theme may
/// ask for a taller band or a bigger mark and can no longer ask for a
/// smaller one.
pub fn strip_geometry(theme: &FooterTheme, image_width: u32, image_height: u32) -> (u32, u32) {
    let by_width = (image_width as f32) * STRIP_MIN_PX_AT_1800 / 1800.0;
    let by_height = (image_height as f32) * STRIP_MIN_HEIGHT_FRACTION;
    let strip = by_width
        .max(by_height)
        .ceil()
        .max(theme.height as f32)
        .max(16.0) as u32;
    let wordmark = (((strip as f32) * WORDMARK_HEIGHT_FRACTION).round() as u32)
        .max(theme.logo_height)
        .min(strip.saturating_sub(4).max(1));
    (strip, wordmark)
}

/// Compose `image` with the strip below it.  Returns the taller canvas.
pub fn compose(image: &RgbaImage, theme: &FooterTheme, fields: &FooterFields) -> RgbaImage {
    let width = image.width();
    let height = image.height();
    let (strip, wordmark_height) = strip_geometry(theme, width, height);
    let mut out = RgbaImage::from_pixel(width, height + strip, theme.background.to_image_rgba());
    for y in 0..height {
        for x in 0..width {
            out.put_pixel(x, y, *image.get_pixel(x, y));
        }
    }

    if let Some(rule) = theme.rule {
        let ink = rule.to_image_rgba();
        for x in 0..width {
            out.put_pixel(x, height, ink);
        }
    }

    let pad = (strip as f32 * 0.28).round().max(8.0) as u32;
    let mut text_x = pad as i32;
    if let Some(logo_path) = theme.logo.as_deref() {
        if let Some(logo) = logo_for(logo_path) {
            let target_h = wordmark_height;
            let target_w = ((logo.width() as f32 / logo.height() as f32) * target_h as f32)
                .round()
                .max(1.0) as u32;
            let scaled = resize(logo, target_w, target_h, FilterType::Lanczos3);
            let origin_y = height + (strip.saturating_sub(target_h)) / 2;
            for y in 0..scaled.height() {
                for x in 0..scaled.width() {
                    let px = scaled.get_pixel(x, y).0;
                    if px[3] == 0 {
                        continue;
                    }
                    let dest_x = pad + x;
                    let dest_y = origin_y + y;
                    if dest_x >= width || dest_y >= out.height() {
                        continue;
                    }
                    blend(
                        &mut out,
                        dest_x,
                        dest_y,
                        Rgba::with_alpha(px[0], px[1], px[2], px[3]),
                    );
                }
            }
            text_x = (pad + target_w + (pad * 3 / 4).max(8)) as i32;
        }
    }

    let title = render_template(&theme.title_template, fields);
    let caption = render_template(&theme.caption_template, fields);
    let (title_factor, caption_factor) = text_factors(strip);
    let title_h = text::bold_line_height_with_factor(1, title_factor);
    let caption_h = text::regular_line_height_with_factor(1, caption_factor);
    let gap = (strip as f32 * 0.06).round().max(2.0) as u32;
    let block = match (title.is_empty(), caption.is_empty()) {
        (false, false) => title_h + gap + caption_h,
        (false, true) => title_h,
        (true, false) => caption_h,
        (true, true) => 0,
    };
    let mut y = height as i32 + ((strip.saturating_sub(block)) / 2) as i32;
    if !title.is_empty() {
        text::draw_text_bold_with_factor(
            &mut out,
            &title,
            text_x,
            y,
            theme.title_ink,
            1,
            title_factor,
        );
        y += (title_h + gap) as i32;
    }
    if !caption.is_empty() {
        text::draw_text_with_factor(&mut out, &caption, text_x, y, theme.ink, 1, caption_factor);
    }
    out
}

#[cfg(test)]
mod strip_tests {
    use super::*;

    fn theme(height: u32, logo_height: u32) -> FooterTheme {
        FooterTheme {
            height,
            background: Rgba::new(0, 0, 0),
            rule: None,
            logo: None,
            logo_height,
            title_template: "{product_title}".to_string(),
            caption_template: "valid {valid_time} | {mesh_or_grid} | {leg} | {note}".to_string(),
            title_ink: Rgba::new(255, 255, 255),
            ink: Rgba::new(200, 200, 200),
        }
    }

    /// WHAT BREAKAGE THIS PREVENTS (gate law): the theme named
    /// ONE pixel height and every sheet got it, so a 64-pixel band with
    /// small type hung under an 1800-pixel delivery.
    #[test]
    fn the_strip_is_a_share_of_the_image_never_a_fixed_sixty_four_pixels() {
        let small = theme(64, 24);
        for (width, height) in [(1800u32, 1464u32), (2400, 1200), (1800, 900)] {
            let (strip, wordmark) = strip_geometry(&small, width, height);
            assert!(
                strip as f32 >= height as f32 * 0.07,
                "{width}x{height}: strip {strip} is under 7 % of the height"
            );
            assert!(
                strip as f32 >= width as f32 * 110.0 / 1800.0,
                "{width}x{height}: strip {strip} is under 110 px per 1800 px of width"
            );
            assert!(strip > small.height, "{width}x{height}: {strip}");
            let ratio = wordmark as f32 / strip as f32;
            assert!(
                (0.60..=0.80).contains(&ratio),
                "{width}x{height}: the wordmark is {ratio} of the strip"
            );
            assert!(wordmark + 4 <= strip);
        }
        // At exactly 1800 wide the floor is the stated 110 px.
        assert_eq!(strip_geometry(&small, 1800, 1000).0, 110);
    }

    /// The theme keeps a say: it may ask for MORE band or MORE mark, never
    /// less, because too little of both is the defect being fixed.
    #[test]
    fn the_themes_own_values_survive_as_floors() {
        let tall = theme(220, 190);
        let (strip, wordmark) = strip_geometry(&tall, 1800, 900);
        assert_eq!(strip, 220);
        assert_eq!(wordmark, 190);
        let (strip, wordmark) = strip_geometry(&theme(16, 1), 900, 600);
        assert_eq!(strip, 55);
        assert_eq!(wordmark, 39);
    }

    /// A taller strip carries bigger type, not the same small type with
    /// more air around it.
    #[test]
    fn the_caption_type_grows_with_the_strip() {
        let (small_title, small_caption) = text_factors(64);
        let (big_title, big_caption) = text_factors(147);
        assert!(big_title > small_title, "{big_title} vs {small_title}");
        assert!(big_caption > small_caption);
        // 147 px of strip: a title of about 38 px and a caption of about
        // 28 px, which reads at arm's length on an 1800 px sheet.
        assert!(big_title * 15.0 >= 34.0, "{}", big_title * 15.0);
        assert!(big_caption * 12.0 >= 24.0, "{}", big_caption * 12.0);
    }

    /// Both rows are still drawn, and the strip is appended below the panel
    /// without moving a pixel of it.
    #[test]
    fn compose_appends_the_strip_and_leaves_the_panel_alone() {
        let panel = RgbaImage::from_pixel(1800, 900, image::Rgba([12, 34, 56, 255]));
        let fields = FooterFields {
            product_title: Some("agent section".to_string()),
            valid_time: Some("12/25 03:00Z".to_string()),
            mesh_or_grid: Some("d02 1 km".to_string()),
            leg: Some("seeded".to_string()),
            note: None,
        };
        let out = compose(&panel, &theme(64, 24), &fields);
        let (strip, _) = strip_geometry(&theme(64, 24), 1800, 900);
        assert_eq!(out.width(), 1800);
        assert_eq!(out.height(), 900 + strip);
        assert_eq!(out.get_pixel(900, 400).0, [12, 34, 56, 255]);
        let ink = (0..out.width())
            .flat_map(|x| (900..out.height()).map(move |y| (x, y)))
            .filter(|(x, y)| out.get_pixel(*x, *y).0 != [0, 0, 0, 255])
            .count();
        assert!(ink > 0, "the strip drew no caption");
    }
}

fn blend(img: &mut RgbaImage, x: u32, y: u32, color: Rgba) {
    let dst = img.get_pixel(x, y).0;
    let alpha = f64::from(color.a) / 255.0;
    let inv = 1.0 - alpha;
    img.put_pixel(
        x,
        y,
        image::Rgba([
            (f64::from(color.r) * alpha + f64::from(dst[0]) * inv).round() as u8,
            (f64::from(color.g) * alpha + f64::from(dst[1]) * inv).round() as u8,
            (f64::from(color.b) * alpha + f64::from(dst[2]) * inv).round() as u8,
            255,
        ]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::{RenderTheme, RenderThemeFile};

    fn fields() -> FooterFields {
        FooterFields {
            product_title: Some("Cloud ice, column maximum, treatment minus control".into()),
            valid_time: Some("2026-08-12 09:00Z".into()),
            mesh_or_grid: Some("3.75 km limited-area hex mesh".into()),
            leg: Some("treatment minus control".into()),
            note: Some("the upper band".into()),
        }
    }

    #[test]
    fn a_template_fills_its_fields_and_drops_the_segments_that_stayed_blank() {
        let template = "valid {valid_time} | {mesh_or_grid} | {leg} | {note}";
        assert_eq!(
            render_template(template, &fields()),
            "valid 2026-08-12 09:00Z | 3.75 km limited-area hex mesh | \
             treatment minus control | the upper band"
        );
        let bare = FooterFields {
            valid_time: Some("2026-08-12 09:00Z".into()),
            ..FooterFields::default()
        };
        assert_eq!(render_template(template, &bare), "valid 2026-08-12 09:00Z");
        let none = FooterFields::default();
        assert_eq!(render_template(template, &none), "");
    }

    #[test]
    fn a_fields_own_separator_stays_inside_its_segment() {
        // The renderer's subtitle row carries separators of its own; before
        // the template was split first, this one field became three
        // segments and the collapse dropped the wrong ones.
        let fields = FooterFields {
            valid_time: Some("A | B | C".into()),
            leg: Some("treatment".into()),
            ..FooterFields::default()
        };
        assert_eq!(
            render_template("valid {valid_time} | {mesh_or_grid} | {leg}", &fields),
            "valid A | B | C | treatment"
        );
    }

    #[test]
    fn a_panel_subtitle_gives_up_only_its_valid_segment() {
        let filled = FooterFields::default().with_panel_defaults(
            Some("2m AGL Temperature"),
            Some("Init 08/25 18Z | F004 | Valid 08/25 22Z | WRF | dx 3 km"),
        );
        assert_eq!(filled.valid_time.as_deref(), Some("08/25 22Z"));
        assert_eq!(filled.product_title.as_deref(), Some("2m AGL Temperature"));
        // A subtitle with no valid segment is kept whole rather than lost.
        let plain = FooterFields::default()
            .with_panel_defaults(None, Some("one line, no segments"));
        assert_eq!(plain.valid_time.as_deref(), Some("one line, no segments"));
    }

    #[test]
    fn a_strip_is_appended_below_the_panel_and_leaves_every_panel_pixel_alone() {
        let file = RenderThemeFile::from_json(
            r##"{"footer": {"height": 48, "background": "#000000", "rule": "#223137",
                            "title": "{product_title}", "caption": "valid {valid_time}",
                            "title_ink": "#e2edf0", "ink": "#94afb8"}}"##,
        )
        .expect("parses");
        let theme = RenderTheme::from_file_spec(&file, None).expect("resolves");
        let footer = theme.footer.expect("the section resolves");
        let panel = RgbaImage::from_pixel(240, 100, image::Rgba([12, 34, 56, 255]));
        let composed = compose(&panel, &footer, &fields());
        assert_eq!(composed.width(), 240);
        assert_eq!(composed.height(), 148, "the strip is 48 px under the panel");
        for y in 0..100u32 {
            for x in 0..240u32 {
                assert_eq!(
                    composed.get_pixel(x, y),
                    panel.get_pixel(x, y),
                    "panel pixel ({x},{y}) moved"
                );
            }
        }
        assert_eq!(
            composed.get_pixel(0, 100).0,
            [0x22, 0x31, 0x37, 255],
            "the rule sits on the strip's first row"
        );
        let ink_pixels = (101..148u32)
            .flat_map(|y| (0..240u32).map(move |x| (x, y)))
            .filter(|(x, y)| composed.get_pixel(*x, *y).0[0] > 40)
            .count();
        assert!(ink_pixels > 50, "the caption drew {ink_pixels} lit pixels");
    }

    #[test]
    fn a_missing_logo_still_draws_the_caption() {
        let footer = FooterTheme {
            height: 40,
            background: Rgba::BLACK,
            rule: None,
            logo: Some(PathBuf::from("no-such-wordmark-file.png")),
            logo_height: 18,
            title_template: "{product_title}".into(),
            caption_template: "{leg}".into(),
            title_ink: Rgba::WHITE,
            ink: Rgba::WHITE,
        };
        let panel = RgbaImage::from_pixel(200, 60, image::Rgba([0, 0, 0, 255]));
        let composed = compose(&panel, &footer, &fields());
        assert_eq!(composed.height(), 100);
        let lit = (60..100u32)
            .flat_map(|y| (0..200u32).map(move |x| (x, y)))
            .filter(|(x, y)| composed.get_pixel(*x, *y).0[0] > 40)
            .count();
        assert!(lit > 50, "the caption drew {lit} lit pixels without a logo");
    }

    #[test]
    fn installed_fields_round_trip_and_clear() {
        set_footer_fields(fields());
        assert_eq!(
            footer_fields().expect("installed").leg.as_deref(),
            Some("treatment minus control")
        );
        clear_footer_fields();
        assert!(footer_fields().is_none());
    }
}
