//! Pair and multi-leg sheets: finished panels of the same product from two
//! or more runs, laid out side by side under one header with a label strip
//! naming each leg.
//!
//! `--pair-sheet REQUEST.json` takes the pairing the caller already made
//! (which file of each leg is the same product) and composes the sheets.
//! Pixels are placed, never recomputed; what this adds over pasting is the
//! layout: under auto layout every panel of one domain has one size and no
//! padding, so a sheet is the panels and a gutter, and the header, labels
//! and margins come from the same layout table as the maps.  Frames wider
//! than the table's bar threshold stack in a column so a pair of wide
//! domains does not become a strip.

use std::fs;
use std::path::{Path, PathBuf};

use image::RgbaImage;
use serde::Deserialize;

const REQUEST_SCHEMA: &str = "arwen.pair-sheet-request.v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    /// The sheet header's title (the comparison's name).
    title: String,
    #[serde(default)]
    subtitle: Option<String>,
    #[serde(default)]
    theme: Option<String>,
    sheets: Vec<SheetRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SheetRequest {
    /// The product the sheet shows: named in errors, never drawn (each
    /// panel carries its own titled header).
    product: String,
    out: PathBuf,
    legs: Vec<Leg>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Leg {
    label: String,
    png: PathBuf,
}

/// Where each leg's panel and label strip go on a sheet of `count` panels
/// of one size.
#[derive(Debug, Clone, PartialEq)]
pub struct PairLayout {
    pub canvas_w: u32,
    pub canvas_h: u32,
    pub header_h: u32,
    /// `(label strip, panel)` origins, one per leg.
    pub cells: Vec<((u32, u32), (u32, u32))>,
    pub label_h: u32,
    pub stacked: bool,
}

/// Lay `count` panels of `panel_w x panel_h` out: in a row, or in a column
/// when a panel is at least as wide as the table's bottom-bar threshold.
pub fn pair_layout(count: u32, panel_w: u32, panel_h: u32, subtitle: bool) -> PairLayout {
    let table = rustwx_render::LayoutTable::builtin();
    let stacked = panel_w as f64 / panel_h.max(1) as f64 >= table.bar.bottom_min_aspect;
    leg_layout(count, panel_w, panel_h, subtitle, stacked)
}

/// Lay `count` panels out in whichever of a row or a column gives the
/// sheet aspect nearest the sheet table's `target_aspect` (a row on a
/// tie).  Three wide panels in a column make a sheet taller than two
/// screens, and three square ones in a column one far taller than wide;
/// the target keeps a sheet of any count screen-shaped.
pub fn sheet_layout(count: u32, panel_w: u32, panel_h: u32, subtitle: bool) -> PairLayout {
    let target = rustwx_render::LayoutTable::builtin().sheet.target_aspect.max(0.1);
    let score = |layout: &PairLayout| {
        (layout.canvas_w as f64 / layout.canvas_h.max(1) as f64 / target)
            .ln()
            .abs()
    };
    let row = leg_layout(count, panel_w, panel_h, subtitle, false);
    let column = leg_layout(count, panel_w, panel_h, subtitle, true);
    if score(&column) < score(&row) { column } else { row }
}

/// Lay `count` panels out in a row, or in a column when `stacked`.
pub fn leg_layout(
    count: u32,
    panel_w: u32,
    panel_h: u32,
    subtitle: bool,
    stacked: bool,
) -> PairLayout {
    let table = rustwx_render::LayoutTable::builtin();
    let margin = table.margin;
    let gutter = table.sheet.gutter;
    let label_h = table.sheet.label_h + 6;
    // One header row, and a second only for a subtitle: each panel carries
    // its own title, times and provenance already.
    let header_h = table.header.pad_top
        + table.header.title_row_h
        + if subtitle { table.header.meta_row_h } else { 0 }
        + table.header.pad_bottom;
    let count = count.max(1);
    let (canvas_w, canvas_h) = if stacked {
        (
            2 * margin + panel_w,
            header_h + count * (label_h + panel_h) + (count - 1) * gutter + margin,
        )
    } else {
        (
            2 * margin + count * panel_w + (count - 1) * gutter,
            header_h + label_h + panel_h + margin,
        )
    };
    let cells = (0..count)
        .map(|index| {
            if stacked {
                let y = header_h + index * (label_h + panel_h + gutter);
                ((margin, y), (margin, y + label_h))
            } else {
                let x = margin + index * (panel_w + gutter);
                ((x, header_h), (x, header_h + label_h))
            }
        })
        .collect();
    PairLayout {
        canvas_w,
        canvas_h,
        header_h,
        cells,
        label_h,
        stacked,
    }
}

fn compose(request: &Request, sheet: &SheetRequest) -> Result<RgbaImage, String> {
    if sheet.legs.len() < 2 {
        return Err(format!("{}: a sheet compares two or more legs", sheet.product));
    }
    let legs: Vec<(String, PathBuf)> = sheet
        .legs
        .iter()
        .map(|leg| (leg.label.clone(), leg.png.clone()))
        .collect();
    compose_with(&request.title, request.subtitle.as_deref(), &legs, pair_layout)
}

/// A sheet of finished panels, one per `(label, png)` leg, arranged by
/// [`sheet_layout`]: the run difference's `A | B | A minus B` sheet.
pub fn compose_legs(
    title: &str,
    subtitle: Option<&str>,
    legs: &[(String, PathBuf)],
) -> Result<RgbaImage, String> {
    compose_with(title, subtitle, legs, sheet_layout)
}

fn compose_with(
    title: &str,
    subtitle_text: Option<&str>,
    legs: &[(String, PathBuf)],
    arrange: fn(u32, u32, u32, bool) -> PairLayout,
) -> Result<RgbaImage, String> {
    if legs.is_empty() {
        return Err("a sheet needs at least one panel".into());
    }
    let mut panels = Vec::with_capacity(legs.len());
    for (_, png) in legs {
        let panel = image::open(png)
            .map_err(|err| format!("read {}: {err}", png.display()))?
            .to_rgba8();
        panels.push(panel);
    }
    // One cell size for every leg: the largest panel.  Planned panels of
    // one domain are all one size; a smaller one sits at its cell's origin.
    let panel_w = panels.iter().map(RgbaImage::width).max().unwrap_or(1);
    let panel_h = panels.iter().map(RgbaImage::height).max().unwrap_or(1);
    let subtitle = subtitle_text.is_some_and(|text| !text.trim().is_empty());
    let layout = arrange(panels.len() as u32, panel_w, panel_h, subtitle);
    let background = *panels[0].get_pixel(0, 0);
    let mut canvas = RgbaImage::from_pixel(layout.canvas_w, layout.canvas_h, background);
    let (title_ink, meta_ink) = rustwx_render::chrome_plan::header_inks();
    let table = rustwx_render::LayoutTable::builtin();
    let header_plan = table.plan_header(
        layout.canvas_w,
        layout.canvas_h,
        rustwx_render::SizeClass::Standard,
        1.0,
    );
    let header = rustwx_render::chrome_plan::PlanHeaderText {
        title: title.to_string(),
        // Every panel names its product with units in its own header; the
        // request's product key is a file stem (`d02-1km xsec qcloud
        // decd39a7`), not a title, so the sheet does not print it.
        when: None,
        meta: subtitle_text.map(str::to_string).unwrap_or_default(),
        right: None,
    };
    let mut header_plan = header_plan;
    header_plan.header_rows = 2;
    header_plan.header.h = layout.header_h;
    rustwx_render::chrome_plan::draw_plan_header(&mut canvas, &header_plan, &header, title_ink, meta_ink);
    for (((label, _), panel), ((label_x, label_y), (panel_x, panel_y))) in
        legs.iter().zip(&panels).zip(&layout.cells)
    {
        rustwx_render::chrome_plan::draw_strip_label(
            &mut canvas,
            label,
            *label_x,
            *label_y,
            layout.label_h,
            title_ink,
            table.sheet.leg_label_px,
        );
        image::imageops::replace(&mut canvas, panel, i64::from(*panel_x), i64::from(*panel_y));
    }
    Ok(canvas)
}

fn run(request_path: &Path) -> Result<(), String> {
    let text = fs::read_to_string(request_path)
        .map_err(|err| format!("read {}: {err}", request_path.display()))?;
    let request: Request = serde_json::from_str(&text)
        .map_err(|err| format!("{}: {err}", request_path.display()))?;
    if request.schema != REQUEST_SCHEMA {
        return Err(format!(
            "{}: schema {:?}, expected {REQUEST_SCHEMA}",
            request_path.display(),
            request.schema
        ));
    }
    if request.sheets.is_empty() {
        return Err("a pair-sheet request names no sheet".into());
    }
    if let Some(spec) = request.theme.as_deref() {
        if spec.trim().is_empty() {
            return Err("pair-sheet theme must not be blank".into());
        }
        rustwx_render::install_theme(rustwx_render::RenderTheme::resolve(spec)?)?;
    }
    let mut failed = 0usize;
    for sheet in &request.sheets {
        let result = compose(&request, sheet).and_then(|canvas| {
            if let Some(parent) = sheet.out.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("create {}: {err}", parent.display()))?;
            }
            canvas
                .save(&sheet.out)
                .map_err(|err| format!("write {}: {err}", sheet.out.display()))
        });
        match result {
            Ok(()) => println!("RENDERED {} {}", sheet.product, sheet.out.display()),
            Err(err) => {
                failed += 1;
                eprintln!("FAILED {} {err}", sheet.product);
            }
        }
    }
    if failed > 0 {
        return Err(format!("{failed} of {} sheet(s) failed", request.sheets.len()));
    }
    Ok(())
}

pub fn try_cli(args: &[String]) -> Option<Result<(), String>> {
    if args.first().map(String::as_str) != Some("--pair-sheet") {
        return None;
    }
    Some(match args {
        [_, request] => run(Path::new(request)),
        _ => Err("Use --pair-sheet REQUEST.json".into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_native_pair_sheet_preserves_every_source_pixel_at_its_own_size() {
        let root = std::env::temp_dir().join(format!(
            "rustwx-pair-pixels-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_nanos()));
        fs::create_dir(&root).unwrap();
        let left = RgbaImage::from_fn(31, 24, |x, y| {
            image::Rgba([(x * 7) as u8, (y * 9) as u8, 103, 255])
        });
        let right = RgbaImage::from_fn(39, 28, |x, y| {
            image::Rgba([61, (x * 5) as u8, (y * 7) as u8, 255])
        });
        let left_path = root.join("a.png");
        let right_path = root.join("b.png");
        left.save(&left_path).unwrap();
        right.save(&right_path).unwrap();
        let legs = [("A".into(), left_path.clone()), ("B".into(), right_path.clone())];
        let canvas = compose_with("Paired fields", None, &legs, pair_layout).unwrap();
        let plan = pair_layout(2, 39, 28, false);
        assert_eq!((canvas.width(), canvas.height()), (plan.canvas_w, plan.canvas_h));
        for (panel, (_, (x, y))) in [left, right].iter().zip(&plan.cells) {
            for (px, py, pixel) in panel.enumerate_pixels() {
                assert_eq!(canvas.get_pixel(x + px, y + py), pixel,
                           "source pixel ({px}, {py}) changed in the pair sheet");
            }
        }
        fs::remove_file(left_path).unwrap();
        fs::remove_file(right_path).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn square_panels_pair_in_a_row_and_wide_panels_stack() {
        let square = pair_layout(2, 888, 866, false);
        assert!(!square.stacked);
        assert_eq!(square.cells.len(), 2);
        let ((_, _), (second_x, _)) = square.cells[1];
        assert_eq!(second_x, 16 + 888 + 12);
        assert_eq!(square.canvas_w, 2 * 16 + 2 * 888 + 12);
        let wide = pair_layout(2, 1350, 592, true);
        assert!(wide.stacked);
        assert_eq!(wide.canvas_w, 2 * 16 + 1350);
        let ((_, _), (_, second_y)) = wide.cells[1];
        assert!(second_y > wide.header_h + 592);
    }

    #[test]
    fn a_three_panel_sheet_takes_the_arrangement_nearest_the_target_aspect() {
        // Three square panels: a row (2.9:1) is nearer 1.5 than a column.
        let square = sheet_layout(3, 888, 866, false);
        assert!(!square.stacked);
        assert_eq!(square.canvas_w, 2 * 16 + 3 * 888 + 2 * 12);
        // Two very wide panels: a column, as the pair rule stacks them.
        let wide = sheet_layout(2, 1350, 592, false);
        assert!(wide.stacked);
        // The pair rule is unchanged by the sheet rule.
        assert_eq!(pair_layout(2, 1350, 592, true).stacked, true);
    }

    #[test]
    fn a_request_needs_two_legs_and_its_own_schema() {
        let text = r#"{"schema":"arwen.pair-sheet-request.v1","title":"t","sheets":[{"product":"p","out":"o.png","legs":[{"label":"a","png":"a.png"}]}]}"#;
        let request: Request = serde_json::from_str(text).unwrap();
        assert!(compose(&request, &request.sheets[0]).is_err());
        assert!(try_cli(&["--help".into()]).is_none());
        assert!(try_cli(&["--pair-sheet".into()]).unwrap().is_err());
    }
}
