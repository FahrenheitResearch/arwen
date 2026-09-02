//! `rw_obsgrid` -- gridded radar OBSERVATIONS through the production
//! render path.
//!
//! The observation file is already classic NetCDF on the model mass grid
//! (`gpuwm-obs.radar-grid`, v1 and v2), so this is a read problem and not a write
//! problem -- and materialising it as a fake wrfout would make the
//! renderer's `TitleProvenance::LocalImport` state that a model produced
//! observations.  Site markers and range rings ride on the same
//! `--overlays` seam every other panel uses.
//!
//! ```text
//! rw_obsgrid --obs FILE.nc --out-dir DIR
//!            [--products z-composite,coverage-depth,radar-overlap,vr-lowest]
//!            [--radar INDEX] [--rings-km 50,100,150] [--sites]
//!            [--width N] [--height N] [--source-label TEXT]
//!            [--overlays FILE.json] [--annotate FILE.json]
//! rw_obsgrid --help | --abi
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use rustwx_render::{LegendControls, LegendMode, LevelDensity, RenderDensity};
use rw_wrfbatch::annotate::{LabelSpec, MapOverlays, PanelAnnotations, PointSpec, RingSpec};
use rw_wrfbatch::obs_grid::ObsRadarGrid;
use rw_wrfbatch::panel::{PanelRequest, layout_path, render_panel};
use rw_wrfbatch::scales;

const ABI_MARKER: &str = "gpuwm-rw-obsgrid-products-v1\tz-composite\tcoverage-depth\t\
radar-overlap\tvr-lowest\tradar-contribution\t\
gpuwm-rw-obsgrid-events-v1\tRENDERED\tSKIPPED\tFAILED\t\
gpuwm-rw-obsgrid-vocabulary-v1\tOBSERVED\tSITES";

const ALL_PRODUCTS: [&str; 4] = [
    "z-composite",
    "coverage-depth",
    "radar-overlap",
    "vr-lowest",
];

struct Args {
    obs: PathBuf,
    out_dir: PathBuf,
    products: Vec<String>,
    radar: Option<usize>,
    rings_km: Vec<f64>,
    sites: bool,
    width: u32,
    height: u32,
    source_label: String,
    overlays: Option<MapOverlays>,
    annotations: Option<PanelAnnotations>,
}

fn usage() -> &'static str {
    "usage: rw_obsgrid --obs FILE.nc --out-dir DIR \
[--products z-composite,coverage-depth,radar-overlap,vr-lowest] [--radar INDEX] \
[--rings-km 50,100,150] [--sites] [--width N] [--height N] [--source-label TEXT] \
[--overlays FILE.json] [--annotate FILE.json]\n       rw_obsgrid --help | --abi"
}

enum Invocation {
    Run(Box<Args>),
    Abi,
    Help,
}

fn parse_args() -> Result<Invocation, String> {
    let mut obs = None;
    let mut out_dir = None;
    let mut products = ALL_PRODUCTS.join(",");
    let mut radar: Option<usize> = None;
    let mut rings_km: Vec<f64> = Vec::new();
    let mut sites = false;
    let mut width = 1_200u32;
    let mut height = 900u32;
    let mut source_label = "NEXRAD Level-II (gridded)".to_string();
    let mut overlays_path: Option<PathBuf> = None;
    let mut annotate_path: Option<PathBuf> = None;

    let mut raw = std::env::args().skip(1);
    while let Some(arg) = raw.next() {
        let mut value = || -> Result<String, String> {
            raw.next().ok_or_else(|| format!("{arg} requires a value"))
        };
        match arg.as_str() {
            "--obs" => obs = Some(PathBuf::from(value()?)),
            "--out-dir" => out_dir = Some(PathBuf::from(value()?)),
            "--products" => products = value()?,
            "--radar" => {
                radar = Some(
                    value()?
                        .parse()
                        .map_err(|err| format!("invalid --radar: {err}"))?,
                )
            }
            "--rings-km" => {
                for token in value()?.split(',') {
                    let token = token.trim();
                    if token.is_empty() {
                        continue;
                    }
                    rings_km.push(
                        token
                            .parse()
                            .map_err(|err| format!("invalid --rings-km {token:?}: {err}"))?,
                    );
                }
            }
            "--sites" => sites = true,
            "--width" => {
                width = value()?
                    .parse()
                    .map_err(|err| format!("invalid --width: {err}"))?
            }
            "--height" => {
                height = value()?
                    .parse()
                    .map_err(|err| format!("invalid --height: {err}"))?
            }
            "--source-label" => source_label = value()?,
            "--overlays" => overlays_path = Some(PathBuf::from(value()?)),
            "--annotate" => annotate_path = Some(PathBuf::from(value()?)),
            "--abi" => return Ok(Invocation::Abi),
            "--help" | "-h" => return Ok(Invocation::Help),
            other => return Err(format!("unknown option {other}")),
        }
    }
    let products: Vec<String> = if products.eq_ignore_ascii_case("all") {
        ALL_PRODUCTS.iter().map(|slug| slug.to_string()).collect()
    } else {
        products
            .split(',')
            .map(|token| token.trim().to_ascii_lowercase())
            .filter(|token| !token.is_empty())
            .collect()
    };
    for product in &products {
        if !ALL_PRODUCTS.contains(&product.as_str()) && product != "radar-contribution" {
            return Err(format!(
                "unknown product {product:?}; choose from {}, radar-contribution",
                ALL_PRODUCTS.join(", ")
            ));
        }
    }
    // Rings without sites is a legitimate request (planning circles with
    // no clutter) but sites without rings is the common one, so neither
    // implies the other and both are stated.
    Ok(Invocation::Run(Box::new(Args {
        obs: obs.ok_or("--obs is required")?,
        out_dir: out_dir.ok_or("--out-dir is required")?,
        products,
        radar,
        rings_km,
        sites,
        width,
        height,
        source_label,
        overlays: overlays_path
            .as_deref()
            .map(MapOverlays::load)
            .transpose()?,
        annotations: annotate_path
            .as_deref()
            .map(PanelAnnotations::load)
            .transpose()?,
    })))
}

fn main() -> ExitCode {
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
    let grid = ObsRadarGrid::open(&args.obs)?;
    println!(
        "OBSERVED grid={}x{} levels={} radars={} valid_time={}",
        grid.ny,
        grid.nx,
        grid.levels,
        grid.radars.len(),
        grid.valid_time.as_deref().unwrap_or("unstated")
    );
    for (index, site) in grid.radars.iter().enumerate() {
        println!(
            "SITES\t{index}\t{}\t{:.4}\t{:.4}",
            site.id, site.lat_deg, site.lon_deg
        );
    }

    // Site markers, labels and planning rings become the same overlay
    // structure a hand-written `--overlays` file would carry, so there is
    // one code path for both and a caller can add to them rather than
    // choose between them.
    let mut overlays = args.overlays.clone().unwrap_or_default();
    if args.sites {
        for site in &grid.radars {
            overlays.points.push(PointSpec {
                lat: site.lat_deg,
                lon: site.lon_deg,
                color: "#101418".to_string(),
                radius_px: 7,
                width_px: 2,
                shape: "circle".to_string(),
            });
            overlays.labels.push(LabelSpec {
                lat: site.lat_deg,
                lon: site.lon_deg,
                text: site.id.clone(),
            });
        }
    }
    if !args.rings_km.is_empty() {
        for site in &grid.radars {
            overlays.rings.push(RingSpec {
                lat: site.lat_deg,
                lon: site.lon_deg,
                radii_km: args.rings_km.clone(),
                color: "#40464f".to_string(),
                width: 1,
                segments: 180,
            });
        }
    }
    let overlays = if overlays.is_empty() {
        None
    } else {
        Some(overlays)
    };

    let valid_day = grid
        .valid_time
        .as_deref()
        .and_then(|stamp| stamp.get(..10).map(str::to_string))
        .unwrap_or_else(|| "undated".to_string());
    let legend = LegendControls {
        density: LevelDensity::default(),
        mode: LegendMode::Stepped,
    };
    let mut rendered = 0usize;
    let mut failed = 0usize;

    for product in &args.products {
        let built = match product.as_str() {
            "z-composite" => Ok((
                grid.z_composite(),
                "Observed composite reflectivity".to_string(),
                "dBZ".to_string(),
                scales::reflectivity_scale(),
            )),
            "coverage-depth" => Ok((
                grid.coverage_depth(),
                "Observed levels per column".to_string(),
                "levels".to_string(),
                scales::count_scale(grid.levels),
            )),
            "radar-overlap" => grid.radar_overlap().map(|values| {
                let max = values.iter().copied().fold(0.0f32, f32::max) as usize;
                (
                    values,
                    "Radars contributing velocity per cell".to_string(),
                    "radars".to_string(),
                    scales::count_scale(max.max(1)),
                )
            }),
            "vr-lowest" => grid.vr_lowest(args.radar).map(|values| {
                let half = values
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .fold(0.0f32, |best, value| best.max(value.abs()))
                    as f64;
                (
                    values,
                    match args.radar.and_then(|index| grid.radars.get(index)) {
                        Some(site) => format!(
                            "Observed radial velocity, lowest observed level ({})",
                            site.id
                        ),
                        None => "Observed radial velocity, lowest observed level".to_string(),
                    },
                    "m s-1".to_string(),
                    scales::radial_velocity_scale(half),
                )
            }),
            "radar-contribution" => {
                let index = args.radar.unwrap_or(0);
                grid.radar_contribution(index).map(|values| {
                    (
                        values,
                        format!(
                            "Cells observed by {}",
                            grid.radars
                                .get(index)
                                .map(|site| site.id.as_str())
                                .unwrap_or("this radar")
                        ),
                        "observed".to_string(),
                        scales::count_scale(1),
                    )
                })
            }
            other => Err(format!("unknown product {other}")),
        };

        let (values, title, units, scale) = match built {
            Ok(built) => built,
            Err(reason) => {
                // An honest skip, not a failure: a single-radar file has
                // no overlap to draw and saying so is the answer.
                println!("SKIPPED {product} {reason}");
                continue;
            }
        };

        let stem = format!(
            "arwen_obs_{}_{}",
            product.replace('-', "_"),
            valid_day.replace('-', "")
        );
        let out_path = layout_path(
            &args.out_dir,
            "obs-grid",
            &product.replace('-', "_"),
            &valid_day,
            &stem,
        );
        let outcome = render_panel(PanelRequest {
            lat_deg: &grid.lat_deg,
            lon_deg: &grid.lon_deg,
            // The observation file carries `XLAT`/`XLONG` and no
            // projection record, so the renderer infers the frame from the
            // mesh exactly as it does for a wrfout with no projection.
            projection: None,
            ny: grid.ny,
            nx: grid.nx,
            values,
            product_slug: product.replace('-', "_"),
            title,
            display_units: units,
            scale,
            cbar_tick_step: None,
            legend,
            render_density: RenderDensity::default(),
            subtitle_left: format!(
                "valid {} | {} radar(s) | {} level(s)",
                grid.valid_time.as_deref().unwrap_or("unstated"),
                grid.radars.len(),
                grid.levels
            ),
            subtitle_center: None,
            subtitle_right: format!("source: {}", args.source_label),
            width: args.width,
            height: args.height,
            contours: Vec::new(),
            colorbar: true,
            overlays: overlays.as_ref(),
            annotations: args.annotations.as_ref(),
            out_path,
        });
        match outcome {
            Ok(path) => {
                rendered += 1;
                println!("RENDERED {product} {}", path.display());
            }
            Err(message) => {
                failed += 1;
                eprintln!("FAILED {product} {message}");
            }
        }
    }

    println!("FINISHED rendered={rendered} failed={failed}");
    if rendered == 0 || failed > 0 {
        return Err(format!(
            "obs grid batch incomplete: rendered={rendered} failed={failed}"
        ));
    }
    Ok(())
}
