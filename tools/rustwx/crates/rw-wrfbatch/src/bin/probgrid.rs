//! `rw_probgrid` -- a precomputed probability field through the production
//! render path.
//!
//! A post-processed probability (a calibrated ML severe-report probability,
//! for one) is not a model field, so `rw_wrfbatch`'s stored-variable lane
//! can only wear its auto-ranged generic ramp, which changes meaning from
//! panel to panel.  This binary reads the 2-D planes a caller names from a
//! classic or NetCDF-4 file carrying `XLAT`/`XLONG` on the mass grid and
//! draws each on a FIXED ladder through the same map, basemap, design and
//! PNG writer every other panel uses.  Observed events (storm reports) ride
//! the `--overlays` points every panel already takes.
//!
//! ```text
//! rw_probgrid --grid FILE.nc --out-dir DIR --vars NAME[,NAME...]
//!             [--scale severe|low|probability] [--domain TOKEN]
//!             [--width N] [--height N] [--source-label TEXT]
//!             [--overlays FILE.json] [--annotate FILE.json]
//! rw_probgrid --help | --abi
//! ```
//!
//! Each variable is a fraction in `[0, 1]` (a leading length-1 time axis is
//! accepted).  `--scale low` is the ladder for probabilities that live under
//! the outlook's first step (an hourly tornado probability).  Its `long_name` attribute is the panel title; the global
//! `valid_time` attribute (ISO UTC) dates the panel and names the PNG.

use std::path::PathBuf;

/// The source-revision stamp every bundled binary carries, so a battery or release check can tell a build of this
/// tree from a stale one.
pub static SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
use std::process::ExitCode;

use rustwx_render::{
    Color, ColorScale, DiscreteColorScale, ExtendMode, LegendControls, LegendMode, LevelDensity,
    RenderDensity,
};
use rw_wrfbatch::annotate::{MapOverlays, PanelAnnotations};
use rw_wrfbatch::panel::{PanelRequest, layout_path, render_panel};
use rw_wrfbatch::scales;

const ABI_MARKER: &str = "gpuwm-rw-probgrid-products-v1\tsevere\tprobability\tlow\t\
gpuwm-rw-probgrid-events-v1\tRENDERED\tSKIPPED\tFAILED";

/// SPC probabilistic-outlook steps in percent: 2, 5, 10, 15, 30, 45, 60.
/// Below 2 % is left transparent, so the map shows where the model said
/// something.  The colours are the outlook's own, so a forecaster reads a
/// panel the way they read the day's outlook.
const SEVERE_LEVELS: [f64; 8] = [2.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 100.0];
const SEVERE_COLORS: [[u8; 3]; 7] = [
    [0, 139, 0],
    [139, 71, 38],
    [255, 200, 0],
    [255, 0, 0],
    [255, 0, 255],
    [145, 44, 238],
    [16, 78, 139],
];

fn severe_scale() -> ColorScale {
    // The renderer spreads a discrete scale's bins evenly along its colourbar and value axis, so the ladder is
    // written as 1 % bins from 2 to 75 %, each wearing its outlook category's colour; above 75 % takes the last.
    let levels: Vec<f64> = (2..=75).map(f64::from).collect();
    let colors = levels[..levels.len() - 1]
        .iter()
        .map(|lo| {
            let k = SEVERE_LEVELS.iter().rposition(|edge| *edge <= *lo).unwrap_or(0).min(SEVERE_COLORS.len() - 1);
            let [r, g, b] = SEVERE_COLORS[k];
            Color::rgba(r, g, b, 255)
        })
        .collect();
    ColorScale::Discrete(DiscreteColorScale {
        levels,
        colors,
        extend: ExtendMode::Max,
        mask_below: Some(SEVERE_LEVELS[0]),
    })
}

/// The low-range ladder in percent: 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30.  An
/// hourly report probability lives under the outlook's first step: Updraft
/// v1's hourly tornado probabilities stay under ~5 %, so the outlook ladder
/// drew a day's hourly tornado sheet as one flat 2-5 % colour or as nothing
/// (2026-10-27, UPDRAFT RESULT.md).  Below 0.5 % is transparent.  Its
/// colours are a sequential ramp of its own, never the outlook's, so a
/// low-range panel cannot be read as an SPC category.
const LOW_LEVELS: [f64; 11] = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 100.0];
const LOW_COLORS: [[u8; 3]; 10] = [
    [198, 219, 239],
    [107, 174, 214],
    [33, 113, 181],
    [116, 196, 118],
    [35, 139, 69],
    [254, 217, 118],
    [253, 141, 60],
    [227, 26, 28],
    [165, 15, 21],
    [122, 1, 119],
];

fn low_scale() -> ColorScale {
    // Unequal steps: drawn with the threshold legend (`LegendMode::Thresholds`), which keeps one colour per step and
    // gives every step the same share of the colourbar, labelled at its edges.
    ColorScale::Discrete(DiscreteColorScale {
        levels: LOW_LEVELS.to_vec(),
        colors: LOW_COLORS.iter().map(|[r, g, b]| Color::rgba(*r, *g, *b, 255)).collect(),
        extend: ExtendMode::Max,
        mask_below: Some(LOW_LEVELS[0]),
    })
}

struct Args {
    grid: PathBuf,
    out_dir: PathBuf,
    vars: Vec<String>,
    scale: String,
    domain: String,
    width: u32,
    height: u32,
    source_label: String,
    overlays: Option<MapOverlays>,
    annotations: Option<PanelAnnotations>,
}

enum Invocation {
    Run(Box<Args>),
    Abi,
    Help,
}

fn usage() -> &'static str {
    "usage: rw_probgrid --grid FILE.nc --out-dir DIR --vars NAME[,NAME...] \
[--scale severe|low|probability] [--domain TOKEN] [--width N] [--height N] [--source-label TEXT] \
[--overlays FILE.json] [--annotate FILE.json]\n       rw_probgrid --help | --abi"
}

fn parse_args() -> Result<Invocation, String> {
    let mut grid = None;
    let mut out_dir = None;
    let mut vars = String::new();
    let mut scale = "severe".to_string();
    let mut domain = "native_grid".to_string();
    let mut width = 1200u32;
    let mut height = 900u32;
    let mut source_label = "WOOF".to_string();
    let mut overlays_path: Option<PathBuf> = None;
    let mut annotate_path: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--grid" => grid = Some(PathBuf::from(value()?)),
            "--out-dir" => out_dir = Some(PathBuf::from(value()?)),
            "--vars" => vars = value()?,
            "--scale" => scale = value()?,
            "--domain" => domain = value()?,
            "--width" => width = value()?.parse().map_err(|err| format!("invalid --width: {err}"))?,
            "--height" => height = value()?.parse().map_err(|err| format!("invalid --height: {err}"))?,
            "--source-label" => source_label = value()?,
            "--overlays" => overlays_path = Some(PathBuf::from(value()?)),
            "--annotate" => annotate_path = Some(PathBuf::from(value()?)),
            "--abi" => return Ok(Invocation::Abi),
            "--help" | "-h" => return Ok(Invocation::Help),
            other => return Err(format!("unknown option {other}")),
        }
    }
    if !matches!(scale.as_str(), "severe" | "low" | "probability") {
        return Err(format!("--scale {scale:?}: choose severe, low or probability"));
    }
    let vars: Vec<String> = vars
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .collect();
    if vars.is_empty() {
        return Err("--vars names no variable".to_string());
    }
    Ok(Invocation::Run(Box::new(Args {
        grid: grid.ok_or("--grid is required")?,
        out_dir: out_dir.ok_or("--out-dir is required")?,
        vars,
        scale,
        domain,
        width,
        height,
        source_label,
        overlays: overlays_path.as_deref().map(MapOverlays::load).transpose()?,
        annotations: annotate_path.as_deref().map(PanelAnnotations::load).transpose()?,
    })))
}

fn string_attribute(nc: &netcrust::File, name: &str) -> Option<String> {
    nc.attribute(name)
        .and_then(|attr| attr.as_string().map(str::to_string))
}

fn var_string_attribute(nc: &netcrust::File, var: &str, name: &str) -> Option<String> {
    nc.variable(var)
        .and_then(|v| v.attribute(name).and_then(|attr| attr.as_string().map(str::to_string)))
}

fn main() -> ExitCode {
    std::hint::black_box(SOURCE_REV_STAMP);
    match parse_args() {
        Ok(Invocation::Abi) => {
            println!("{ABI_MARKER}");
            ExitCode::SUCCESS
        }
        Ok(Invocation::Help) => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        Ok(Invocation::Run(args)) => match run(*args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("{message}");
                ExitCode::FAILURE
            }
        },
        Err(message) => {
            eprintln!("{message}");
            eprintln!("{}", usage());
            ExitCode::from(2)
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    let path = &args.grid;
    let nc = netcrust::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let lat = nc
        .read_array_f64("XLAT")
        .map_err(|err| format!("{}: read XLAT: {err}", path.display()))?;
    let lon = nc
        .read_array_f64("XLONG")
        .map_err(|err| format!("{}: read XLONG: {err}", path.display()))?;
    let shape = lat.shape().to_vec();
    if shape.len() < 2 || lon.shape() != lat.shape() {
        return Err(format!(
            "{}: XLAT {:?} / XLONG {:?} are not one mass grid",
            path.display(),
            lat.shape(),
            lon.shape()
        ));
    }
    let ny = shape[shape.len() - 2];
    let nx = shape[shape.len() - 1];
    let points = ny * nx;
    let lat_deg: Vec<f32> = lat.values()[..points].iter().map(|v| *v as f32).collect();
    let lon_deg: Vec<f32> = lon.values()[..points].iter().map(|v| *v as f32).collect();
    let valid = string_attribute(&nc, "valid_time").unwrap_or_else(|| "unstated".to_string());
    let valid_day = valid.get(..10).unwrap_or("undated").to_string();
    let stamp: String = valid.chars().filter(|c| c.is_ascii_digit()).take(12).collect();
    let lead = string_attribute(&nc, "lead").unwrap_or_default();
    let legend = LegendControls {
        density: LevelDensity::default(),
        mode: if args.scale == "low" { LegendMode::Thresholds } else { LegendMode::Stepped },
    };
    let (scale, units, factor) = match args.scale.as_str() {
        "severe" => (severe_scale(), "%".to_string(), 100.0f32),
        "low" => (low_scale(), "%".to_string(), 100.0f32),
        _ => (scales::probability_scale(), "probability".to_string(), 1.0f32),
    };
    let mut rendered = 0usize;
    let mut failed = 0usize;
    for var in &args.vars {
        let values = match nc.read_array_f64(var) {
            Ok(array) if array.values().len() >= points => array.values()[array.values().len() - points..]
                .iter()
                .map(|v| if v.is_finite() { *v as f32 * factor } else { f32::NAN })
                .collect::<Vec<f32>>(),
            Ok(array) => {
                println!("SKIPPED {var} carries {} value(s), not one {ny}x{nx} plane", array.values().len());
                continue;
            }
            Err(err) => {
                println!("SKIPPED {var} {err}");
                continue;
            }
        };
        let title = var_string_attribute(&nc, var, "long_name").unwrap_or_else(|| var.clone());
        let slug = var.to_ascii_lowercase();
        let stem = format!("woof_{slug}_valid_{}_{}00z_lead_{}", &stamp.get(..8).unwrap_or("00000000"),
                           &stamp.get(8..12).unwrap_or("0000"), if lead.is_empty() { "na" } else { &lead });
        let out_path = layout_path(&args.out_dir, &args.domain, &slug, &valid_day, &stem);
        let outcome = render_panel(PanelRequest {
            lat_deg: &lat_deg,
            lon_deg: &lon_deg,
            projection: None,
            ny,
            nx,
            values,
            product_slug: slug.clone(),
            title,
            display_units: units.clone(),
            scale: scale.clone(),
            cbar_tick_step: None,
            legend,
            // The outlook and low ladders are discrete: no interpolated sub-levels or blended colours between steps.
            render_density: if args.scale != "probability" {
                RenderDensity { fill: LevelDensity::default(), palette_multiplier: 1 }
            } else {
                RenderDensity::default()
            },
            subtitle_left: format!("valid {valid}{}", if lead.is_empty() { String::new() } else { format!(" | +{lead} h") }),
            subtitle_center: None,
            subtitle_right: format!("source: {}", args.source_label),
            width: args.width,
            height: args.height,
            contours: Vec::new(),
            colorbar: true,
            overlays: args.overlays.as_ref(),
            annotations: args.annotations.as_ref(),
            out_path,
        });
        match outcome {
            Ok(path) => {
                rendered += 1;
                println!("RENDERED {var} {}", path.display());
            }
            Err(message) => {
                failed += 1;
                eprintln!("FAILED {var} {message}");
            }
        }
    }
    println!("FINISHED rendered={rendered} failed={failed}");
    if rendered == 0 || failed > 0 {
        return Err(format!("probability batch incomplete: rendered={rendered} failed={failed}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_render::{ColormapBuildOptions, Rgba, build_colormap};

    fn colour(scale: &ColorScale, mode: LegendMode, percent: f64) -> Rgba {
        // The fill density run() gives both discrete ladders: no interpolated sub-levels.
        let options = ColormapBuildOptions {
            legend: LegendControls { density: LevelDensity::default(), mode },
            render_density: RenderDensity { fill: LevelDensity::default(), palette_multiplier: 1 },
        };
        build_colormap(scale, options).map(percent)
    }

    /// The breakage: on the outlook ladder 0.8 % is not drawn and 2.5 % and 4.5 % wear one colour, so an hourly
    /// tornado sheet shows nothing or one flat blob. On the low ladder each step is drawn in its own colour.
    #[test]
    fn low_ladder_separates_hourly_probabilities_the_outlook_ladder_merges() {
        let outlook = severe_scale();
        assert_eq!(colour(&outlook, LegendMode::Stepped, 0.8), Rgba::TRANSPARENT);
        assert_eq!(colour(&outlook, LegendMode::Stepped, 2.5), colour(&outlook, LegendMode::Stepped, 4.5));
        let low = low_scale();
        let seen: Vec<Rgba> = [0.8, 1.5, 2.5, 4.5, 6.0, 8.0, 12.0, 17.0, 25.0, 40.0]
            .iter()
            .map(|p| colour(&low, LegendMode::Thresholds, *p))
            .collect();
        for (i, c) in seen.iter().enumerate() {
            assert_ne!(*c, Rgba::TRANSPARENT, "step {i} not drawn");
            assert!(!seen[..i].contains(c), "step {i} shares a colour");
        }
        assert_eq!(colour(&low, LegendMode::Thresholds, 0.3), Rgba::TRANSPARENT);
    }
}
