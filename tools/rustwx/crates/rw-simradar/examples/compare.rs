//! Render simulated and observed radar with identical Rust map presentation.
#[path = "../src/manifest.rs"]
mod manifest;
#[path = "../src/ppi.rs"]
mod ppi;
use std::path::Path;
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: compare SIMULATED_LEVEL2 OBSERVED_LEVEL2 OUTPUT_DIRECTORY [OBSERVED_SWEEP_INDEX]".into());
    }
    let simulated =
        nexrad_io::decode_volume_from_path(Path::new(&args[0])).map_err(|e| e.to_string())?;
    let mut observed =
        nexrad_io::decode_volume_from_path(Path::new(&args[1])).map_err(|e| e.to_string())?;
    let observation_volume_time = observed.volume_time;
    let mut selected_sweep = None;
    if let Some(index) = args.get(3) {
        let index: usize = index.parse().map_err(|_| "invalid observed sweep index")?;
        let mut cut = observed
            .cuts
            .get(index)
            .ok_or("observed sweep is absent")?
            .clone();
        let times: Vec<_> = cut
            .radials
            .iter()
            .map(|r| {
                bowecho_simradar::wrf_radar_validation::radial_acquisition_time_utc(&observed, r)
                    .ok_or("observed ray time is unavailable")
            })
            .collect::<Result<_, _>>()?;
        let first = *times.iter().min().ok_or("observed sweep has no rays")?;
        let last = *times.iter().max().unwrap();
        cut.elevation_deg =
            cut.radials.iter().map(|r| r.elevation_deg).sum::<f32>() / cut.radials.len() as f32;
        observed.volume_time = first;
        selected_sweep = Some(
            serde_json::json!({"index":index,"mean_elevation_deg":cut.elevation_deg,
            "start":observed.volume_time,"end":last}),
        );
        observed.cuts = vec![cut];
    }
    if simulated.site.id != observed.site.id {
        return Err("comparison radar sites differ".into());
    }
    let root = Path::new(&args[2]);
    let model = ppi::render_labeled(
        root,
        "d01",
        "model",
        &simulated,
        1,
        960,
        900,
        "WOOF simulated",
        Some(230_000.0),
    )?;
    let truth = ppi::render_labeled(
        root,
        "d01",
        "observed",
        &observed,
        2,
        960,
        900,
        "Observed",
        Some(230_000.0),
    )?;
    let mut products = Vec::new();
    for left in &model {
        if let Some(right) = truth.iter().find(|i| i.field == left.field) {
            let l = image::open(root.join(&left.artifact.path))
                .map_err(|e| e.to_string())?
                .to_rgba8();
            let r = image::open(root.join(&right.artifact.path))
                .map_err(|e| e.to_string())?
                .to_rgba8();
            let mut sheet = image::RgbaImage::from_pixel(
                l.width() + r.width() + 16,
                l.height().max(r.height()),
                image::Rgba([245, 247, 249, 255]),
            );
            image::imageops::overlay(&mut sheet, &l, 0, 0);
            image::imageops::overlay(&mut sheet, &r, i64::from(l.width() + 16), 0);
            let path = root.join(format!(
                "{}_{}_comparison.png",
                simulated.site.id, left.field
            ));
            sheet.save(&path).map_err(|e| e.to_string())?;
            products.push(manifest::artifact(root, &path, "png")?);
        }
    }
    let receipt = serde_json::json!({"site":simulated.site.id,"model_time":simulated.volume_time,
        "observation_volume_time":observation_volume_time,"observation_time":observed.volume_time,
        "observed_sweep":selected_sweep,"observation_offset_seconds":(observed.volume_time-simulated.volume_time).num_milliseconds() as f64*0.001,
        "simulated_sha256":manifest::file_hash(Path::new(&args[0]))?.0,
        "observed_sha256":manifest::file_hash(Path::new(&args[1]))?.0,
        "renderer":"rw_wrfbatch::panel::render_panel","products":products});
    manifest::atomic_bytes(
        &root.join("comparison.json"),
        &serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )?;
    println!("{receipt}");
    Ok(())
}
