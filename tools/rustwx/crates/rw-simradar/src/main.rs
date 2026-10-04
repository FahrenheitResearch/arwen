//! Native WOOF simulated-radar history consumer.
mod adapter;
mod atmosphere;
mod input_contract;
mod loops;
mod manifest;
mod ppi;
mod request;
mod resources;

use bowecho_simradar::radar_core::{MomentType, RadarVolume};
use bowecho_simradar::wrf_scene_adapter::inventory_wrf_paths;
use bowecho_simradar::wrf_scene_inventory::WrfSceneLocator;
use bowecho_simradar::wrf_temporal::{TemporalSamplingOutcome, plan_for_scene};
use bowecho_simradar::{
    WrfFile, WrfRadarFields, build_synthetic_volume_reporting_temporal,
    read_wrf_radar_fields_for_config, try_build_synthetic_volume,
};
use request::{Fields, Request};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// PPI images drawn per volume: the lowest distinct tilts, each in pixels.
/// `resources::output_bound` prices exactly these.
const PPI_TILTS: usize = 2;
const PPI_WIDTH: u32 = 960;
const PPI_HEIGHT: u32 = 900;

fn read_scene(
    scene: &WrfSceneLocator,
    config: &bowecho_simradar::SyntheticRadarConfig,
) -> Result<WrfRadarFields, String> {
    let file = WrfFile::open(&scene.path).map_err(|e| e.to_string())?;
    let mut fields =
        read_wrf_radar_fields_for_config(&file, &scene.source_identity, scene.time_index, config)?;
    atmosphere::native_winds(&file, scene.time_index, &mut fields)?;
    Ok(fields)
}

fn field_name(moment: &MomentType) -> &str {
    match moment {
        MomentType::Reflectivity => "reflectivity",
        MomentType::Velocity => "velocity",
        MomentType::DifferentialReflectivity => "zdr",
        MomentType::CorrelationCoefficient => "rhohv",
        MomentType::DifferentialPhase => "phidp",
        MomentType::SpecificDifferentialPhase => "kdp",
        MomentType::SpectrumWidth => "spectrum_width",
        MomentType::Unknown(v) => v,
    }
}

fn select_fields(volume: &mut RadarVolume, fields: &Fields) -> Result<(), String> {
    if let Fields::List(wanted) = fields {
        for name in wanted {
            if !volume
                .cuts
                .iter()
                .any(|c| c.moments.keys().any(|m| field_name(m) == name))
            {
                return Err(format!(
                    "requested radar field {name} is unavailable for this history's microphysics"
                ));
            }
        }
        for cut in &mut volume.cuts {
            cut.moments
                .retain(|moment, _| wanted.iter().any(|f| f == field_name(moment)));
        }
    } else {
        for cut in &mut volume.cuts {
            cut.moments.retain(|moment, _| {
                matches!(
                    moment,
                    MomentType::Reflectivity
                        | MomentType::Velocity
                        | MomentType::DifferentialReflectivity
                        | MomentType::CorrelationCoefficient
                        | MomentType::DifferentialPhase
                        | MomentType::SpecificDifferentialPhase
                )
            });
        }
    }
    Ok(())
}

fn process(request: Request) -> Result<PathBuf, String> {
    if request.schema != "simulated-radar.request/v1" {
        return Err("unsupported simulated radar request schema".into());
    }
    request.config.validate()?;
    // One request draws one radar look, selected before the first PPI.
    rustwx_render::install_radar_color_set(request.config.color_set()?)?;
    // Reject impossible scans before even hashing the history tape.
    request.config.check_memory(0)?;
    if request.history_paths.is_empty() {
        return Err("simulated radar requires history files".into());
    }
    if !request.scene_shapes.is_empty() {
        return Err("scene_shapes prices a forecast before its history exists (--estimate only); a volume request reads the shapes of its own history files".into());
    }
    let mut histories = request.history_paths;
    histories.sort();
    histories.dedup();
    // The histories that publish volumes; the others are scan-timing
    // neighbours only. A name outside the request would publish nothing.
    let volume_paths: Option<std::collections::BTreeSet<PathBuf>> = match request.volume_paths {
        None => None,
        Some(paths) => {
            if paths.is_empty() {
                return Err("volume_paths is empty, so the request would publish no volume; omit it to publish every history".into());
            }
            if let Some(path) = paths.iter().find(|path| histories.binary_search(*path).is_err()) {
                return Err(format!(
                    "volume_paths names {} which is not in history_paths; a volume is published only from a history the request reads",
                    path.display()
                ));
            }
            Some(paths.into_iter().collect())
        }
    };
    for path in &histories {
        input_contract::validate_history(path)?;
    }
    let source_hashes: BTreeMap<_, _> = histories
        .iter()
        .map(|path| manifest::file_hash(path).map(|(hash, _)| (path.clone(), hash)))
        .collect::<Result<_, _>>()?;
    let inventory = inventory_wrf_paths(&histories)?;
    if inventory
        .inventory
        .groups
        .iter()
        .all(|g| g.scenes.is_empty())
    {
        return Err("history contains no model records to scan".into());
    }
    let mut unique_frames = std::collections::BTreeSet::new();
    for group in &inventory.inventory.groups {
        for scene in &group.scenes {
            if !unique_frames.insert((scene.run_domain.domain, scene.time.valid_time().cloned())) {
                return Err("inputs repeat a domain and valid time across runs or grids; give each run its own radar output directory".into());
            }
        }
    }
    for note in inventory.notes {
        eprintln!("warning: {}: {}", note.source_name, note.message);
    }
    let _lock = manifest::RunLock::acquire(&request.outdir)?;
    let provenance: serde_json::Value =
        serde_json::from_str(include_str!("../../../vendor/bowecho/SOURCE.json"))
            .map_err(|e| e.to_string())?;
    let extraction = provenance
        .get("extraction_commit")
        .and_then(|v| v.as_str())
        .ok_or("vendored BowEcho source lacks extraction commit")?;
    let mut manifest = manifest::Manifest::load(&request.outdir, extraction)?;
    manifest.bowecho_source_commit = provenance["source_commit"]
        .as_str()
        .ok_or("BowEcho source pin is missing")?
        .into();
    manifest.bowecho_extraction_commit = extraction.into();
    #[cfg(target_os = "linux")]
    let executable = PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let binary_hash = manifest::file_hash(&executable)?.0;
    let implementation = serde_json::json!({"bowecho_source_commit":provenance["source_commit"],
        "bowecho_extraction_commit":extraction,"writer_source_commit":"c206a2495c36341caa2a62ff7be3025320dbb028",
        "contract":request::ABI,"native_source_sha256":env!("SIMRADAR_IMPLEMENTATION_SHA256"),
        "native_binary_sha256":binary_hash});
    if !request.config.enabled {
        manifest.save(&request.outdir)?;
        return Ok(manifest::Manifest::path(&request.outdir));
    }
    let config_hash =
        manifest::digest(&serde_json::to_vec(&request.config).map_err(|e| e.to_string())?);
    let base = request.config.native();
    base.validate_science_contract()?;
    let mut processed = 0usize;
    for group in inventory.inventory.groups {
        if !group.diagnostics.duplicate_times.is_empty() {
            return Err("history inputs repeat a domain time; ambiguous radar atmospheres cannot share one volume path".into());
        }
        if !group.diagnostics.unavailable_times.is_empty() {
            return Err("history has no valid time; radar volumes cannot be dated".into());
        }
        let domain = group.key.run_domain.domain.label();
        for (index, scene) in group.scenes.iter().enumerate() {
            if volume_paths
                .as_ref()
                .is_some_and(|selected| !selected.contains(&scene.path))
            {
                continue;
            }
            let started = Instant::now();
            let plan = plan_for_scene(
                &group,
                index,
                chrono::Duration::milliseconds(base.planned_scan_duration_ms()),
                base.atmosphere_time_mode,
                base.missing_neighbor_policy,
            )
            .map_err(|e| e.to_string())?
            .ok_or("radar temporal plan unexpectedly omitted an output")?;
            let mut source_times = vec![plan.anchor_time.to_rfc3339()];
            let mut source_history = vec![source_description(&scene.locator(), &source_hashes)];
            if plan.outcome == TemporalSamplingOutcome::LinearAdjacent {
                if let Some(time) = plan.neighbor_time {
                    source_times.push(time.to_rfc3339());
                }
                if let Some(source) = &plan.neighbor {
                    source_history.push(source_description(source, &source_hashes));
                }
            }
            let generation=manifest::digest(&serde_json::to_vec(&serde_json::json!({
                "config":request.config,"implementation":implementation,"sources":source_history,
                "source_times":source_times,"domain":domain,"run":group.key.run_domain.run.0,
                "timing_outcome":format!("{:?}",plan.outcome),
            })).map_err(|e|e.to_string())?);
            if manifest.complete_generation(
                &request.outdir,
                &domain,
                &plan.anchor_time.to_rfc3339(),
                &generation,
            ) {
                println!(
                    "{}",
                    serde_json::json!({"event":"already_complete","domain":domain,"valid_time":plan.anchor_time.to_rfc3339()})
                );
                continue;
            }
            let header = WrfFile::open(&scene.path).map_err(|e| e.to_string())?;
            let scene_bytes = resources::atmosphere_bytes(header.nx, header.ny, header.nz)?;
            drop(header);
            let resident_scenes = if plan.outcome == TemporalSamplingOutcome::LinearAdjacent {2} else {1};
            request.config.check_memory(scene_bytes.checked_mul(resident_scenes).ok_or("radar memory estimate overflow")?)?;
            let fields = read_scene(&scene.locator(), &base)?;
            let sites = request.config.resolve_sites(&fields)?;
            if sites.is_empty() {
                manifest.retire_where(|v| {
                    v.domain == domain && v.valid_time == plan.anchor_time.to_rfc3339()
                });
                let warning = format!(
                    "no NEXRAD coverage overlaps {domain} at {}; custom sites can scan this domain",
                    plan.anchor_time
                );
                eprintln!("warning: {warning}");
                if !manifest.warnings.contains(&warning) {
                    manifest.warnings.push(warning);
                }
                manifest.save(&request.outdir)?;
                continue;
            }
            let expected_sites: Vec<_> = sites.iter().map(|s| s.id.clone()).collect();
            // Superseded generations leave the manifest here and their files
            // are deleted once a save no longer names them.
            manifest.retire_where(|v| {
                v.domain == domain
                    && v.valid_time == plan.anchor_time.to_rfc3339()
                    && v.generation != generation
            });
            let neighbor = if plan.outcome == TemporalSamplingOutcome::LinearAdjacent {
                request.config.check_memory(scene_bytes)?;
                Some(read_scene(
                    plan.neighbor
                        .as_ref()
                        .ok_or("temporal plan lacks neighbor")?,
                    &base,
                )?)
            } else {
                None
            };
            let read_seconds = started.elapsed().as_secs_f64();
            for site in sites {
                if manifest.site_complete(
                    &request.outdir,
                    &domain,
                    &plan.anchor_time.to_rfc3339(),
                    &generation,
                    &site.id,
                ) {
                    continue;
                }
                request.config.check_memory(0)?;
                let site_started = Instant::now();
                let mut config = base.clone();
                config.site_id = site.id.clone();
                config.site_lat_deg = Some(site.lat);
                config.site_lon_deg = Some(site.lon);
                config.antenna_msl_m = site.height_m;
                let mut volume = if let Some(neighbor) = neighbor.as_ref() {
                    build_synthetic_volume_reporting_temporal(
                        &fields,
                        neighbor,
                        plan.anchor_time,
                        &config,
                        &|_| {},
                        &plan,
                    )?
                } else {
                    try_build_synthetic_volume(&fields, plan.anchor_time, &config)?
                };
                select_fields(&mut volume, &request.config.fields)?;
                // Producer labels are consumer metadata; the forward operator remains unchanged.
                volume.site.name = Some("WOOF simulated radar".into());
                let fm301 = adapter::to_fm301(&volume)?;
                let stem = plan.anchor_time.format("%Y%m%dT%H%M%SZ").to_string();
                let directory = request
                    .outdir
                    .join("radar")
                    .join(&domain)
                    .join(&site.id)
                    .join(&generation);
                let mut files = Vec::new();
                let mut writer_reports = BTreeMap::new();
                for format in &request.config.formats {
                    let suffix = match format.as_str() {
                        "level2" => "ar2v",
                        "cfradial1" => "cfradial1.nc",
                        "cfradial2" => "cfradial2.nc",
                        "odim" => "odim.h5",
                        _ => unreachable!(),
                    };
                    let path = directory.join(format!("{stem}.{suffix}"));
                    let (bytes, report) = adapter::encode(&fm301, format)?;
                    manifest::atomic_bytes(&path, &bytes)?;
                    files.push(manifest::artifact(&request.outdir, &path, format)?);
                    writer_reports.insert(format.clone(), report);
                }
                let images = ppi::render(
                    &request.outdir,
                    &domain,
                    &generation,
                    &volume,
                    PPI_TILTS,
                    PPI_WIDTH,
                    PPI_HEIGHT,
                )?;
                let offsets = volume
                    .cuts
                    .iter()
                    .flat_map(|c| c.radials.iter())
                    .map(|r| r.time_offset_ms);
                let last_ms = offsets.max().unwrap_or(0);
                let mut fields_present: Vec<_> = volume
                    .cuts
                    .iter()
                    .flat_map(|c| c.moments.keys().map(|m| field_name(m).to_owned()))
                    .collect();
                fields_present.sort();
                fields_present.dedup();
                let timing_used = match plan.outcome {
                    TemporalSamplingOutcome::Frozen => "history",
                    TemporalSamplingOutcome::LinearAdjacent => "linear_adjacent",
                    TemporalSamplingOutcome::HeldAnchor(_) => "held_anchor",
                };
                let elapsed = site_started.elapsed().as_secs_f64();
                manifest.upsert(manifest::Volume {
                    domain: domain.clone(),
                    generation: generation.clone(),
                    expected_sites: expected_sites.clone(),
                    implementation: implementation.clone(),
                    site: manifest::Site {
                        id: site.id.clone(),
                        latitude_deg: site.lat,
                        longitude_deg: site.lon,
                        antenna_height_msl_m: f64::from(
                            volume
                                .site
                                .elevation_m
                                .ok_or("simulated radar lacks antenna height")?,
                        ),
                    },
                    valid_time: plan.anchor_time.to_rfc3339(),
                    scan_start: plan.anchor_time.to_rfc3339(),
                    scan_end: (plan.anchor_time
                        + chrono::Duration::milliseconds(i64::from(last_ms)))
                    .to_rfc3339(),
                    timing_requested: request.config.timing.clone(),
                    timing_used: timing_used.into(),
                    source_times: source_times.clone(),
                    source_history: source_history.clone(),
                    tilts_deg: volume.cuts.iter().map(|c| c.elevation_deg).collect(),
                    fields: fields_present,
                    config_sha256: config_hash.clone(),
                    reflectivity_source: fields.ref_source.into(),
                    dual_pol_status: fields.dual_pol_status.clone(),
                    writer_reports,
                    files,
                    images,
                    elapsed_seconds: elapsed,
                });
                manifest.save(&request.outdir)?;
                processed += 1;
                println!(
                    "{}",
                    serde_json::json!({"event":"volume_committed","domain":domain,"site":site.id,
                    "valid_time":plan.anchor_time.to_rfc3339(),"read_seconds":read_seconds,"volume_seconds":elapsed})
                );
            }
        }
    }
    let _ = processed;
    loops::update(&request.outdir, &mut manifest)?;
    Ok(manifest::Manifest::path(&request.outdir))
}

fn source_description(scene: &WrfSceneLocator, hashes: &BTreeMap<PathBuf, String>) -> String {
    format!(
        "{}#{}:sha256={}",
        scene.path.file_name().unwrap_or_default().to_string_lossy(),
        scene.time_index,
        hashes[&scene.path]
    )
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--capabilities" {
        println!("{}", serde_json::json!({
            "schema":"rw-simradar.capabilities/v1",
            "request_abi":request::ABI,
            "canonical_columns_abi":atmosphere::ABI,
            "resource_estimate_schema":resources::SCHEMA,
            "input_validation":"full-columns/v1",
            "memory_policy":"checked-estimate-within-host-and-cgroup-headroom/v1",
            "ppi_fields":ppi::PPI_FIELDS,
            "color_tables":rustwx_render::RadarColorSet::NAMES,
            "site_count_cap":null
        }));
        return Ok(());
    }
    if args.len() == 4 && args[0] == "--canonical-atmosphere" && args[2] == "--out" {
        atmosphere::convert(Path::new(&args[1]), Path::new(&args[3]))?;
        println!(
            "{}",
            serde_json::json!({"event":"canonical_atmosphere_committed", "path":args[3]})
        );
        return Ok(());
    }
    if args.len() == 1 && args[0] == "--abi" {
        println!("{}", request::ABI);
        return Ok(());
    }
    if args.len() == 1 && args[0] == "--canonical-abi" {
        println!("{}", atmosphere::ABI);
        return Ok(());
    }
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!(
            "{}\nrw_simradar --capabilities\nrw_simradar --estimate REQUEST.json schema=simulated-radar.resources/v1\nrw_simradar --canonical-atmosphere SOURCE.nc --out OUTPUT.nc schema=native-atmosphere.columns/v1",
            request::ABI
        );
        return Ok(());
    }
    if args.len() != 2 || (args[0] != "--request" && args[0] != "--estimate") {
        return Err(format!("usage: {}", request::ABI));
    }
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| e.to_string())?;
    }
    let mut document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(Path::new(&args[1])).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if let Some(config) = document.get_mut("config").and_then(|v| v.as_object_mut()) {
        if config.contains_key("elevations_deg") && !config.contains_key("scan_strategy") {
            config.insert("scan_strategy".into(), serde_json::json!("custom"));
        }
        if config.get("scan_strategy").and_then(|v| v.as_str()) == Some("low_tilts")
            && !config.contains_key("elevations_deg")
        {
            config.insert("elevations_deg".into(), serde_json::json!([0.5, 0.9, 1.3]));
        }
    }
    let request: Request = serde_json::from_value(document).map_err(|e| e.to_string())?;
    if args[0] == "--estimate" {
        println!("{}", serde_json::to_string_pretty(&resources::estimate(&request)?).map_err(|e|e.to_string())?);
        return Ok(());
    }
    let manifest = process(request)?;
    println!(
        "{}",
        serde_json::json!({"event":"complete","manifest":manifest})
    );
    Ok(())
}
/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the release cut can prove a staged
/// binary matches the commit being released by reading bytes alone.
/// `build.rs` injects the value; `main` references the constant so the
/// linker cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() -> std::process::ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn ppi_fields_carry_the_manifest_field_names() {
        for &field in crate::ppi::PPI_FIELDS {
            let drawn = crate::ppi::drawn_field(field);
            assert_eq!(crate::field_name(&drawn.moment), field);
        }
    }
}
