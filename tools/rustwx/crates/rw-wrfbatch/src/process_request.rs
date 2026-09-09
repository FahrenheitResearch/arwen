//! One immutable live WRF frame to the original complete native store.
//! The portable companion receipt only describes the existing RWS/RWG codecs.
use crate::viewer_profile::{PROFILE as VIEWER_PROFILE, ProductAvailability, ViewerProfile};
use crate::wrf_process::{
    LiveWrfTarget, WrfProcessMessage, WrfProcessOptions, spawn_process_live_path,
};
use rw_store::{RwsExactTime, grid::GridFile, reader::HourReader, run::RwsRunManifest};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

pub const REQUEST_SCHEMA: &str = "arwen.wrf-process-request.v1";
pub const RESULT_SCHEMA: &str = "arwen.wrf-process-result.v1";
pub const VIEWER_REQUEST_SCHEMA: &str = "arwen.wrf-process-request.v2";
pub const VIEWER_RESULT_SCHEMA: &str = "arwen.wrf-process-result.v2";
const FRAME_SCHEMA: &str = "arwen.companion-store-frame.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessRequest {
    pub schema: String,
    pub path: PathBuf,
    pub source_sha256: String,
    pub case_id: String,
    pub domain: String,
    pub valid_utc: String,
    pub store_root: PathBuf,
    #[serde(default)]
    pub lead_seconds: Option<u64>,
    #[serde(default)]
    pub heavy_ecape: bool,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub products: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameIdentity {
    pub case_id: String,
    pub source: String,
    pub model: String,
    pub member: Option<u16>,
    pub valid_unix: i64,
    pub lead_seconds: u64,
    pub source_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedFrame {
    pub schema: String,
    pub id: String,
    pub identity: FrameIdentity,
    pub store_root: PathBuf,
    pub model_slug: String,
    pub run_slug: String,
    pub storage_slot: u16,
    pub hour_path: PathBuf,
    pub grid_sha256: String,
    pub shape: [usize; 2],
    pub variables: Vec<String>,
    pub levels_hpa: Vec<u16>,
    pub rws_sha256: String,
    pub rws_bytes: u64,
    pub processing_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleFile {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoreMember {
    pub key: String,
    pub relative_path: String,
    pub path: PathBuf,
    pub kind: String,
    pub bytes: u64,
    pub sha256: String,
    pub grid_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessResult {
    pub schema: String,
    pub frame: ProcessedFrame,
    pub domain: String,
    pub grid_path: PathBuf,
    pub run_json_path: PathBuf,
    pub receipt_path: PathBuf,
    pub cache_hit: bool,
    pub notes: Vec<String>,
    pub files: Vec<BundleFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initialization_unix: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub products: Vec<ProductAvailability>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub members: Vec<StoreMember>,
}

/// Separate invocation grammar keeps regular rendering and its ABI unchanged.
pub fn try_cli(args: &[String]) -> Option<Result<(), String>> {
    if !args
        .iter()
        .any(|arg| arg == "--process-request" || arg == "--process-result")
    {
        return None;
    }
    Some((|| {
        let mut input = None;
        let mut output = None;
        let mut index = 0;
        while index < args.len() {
            let target = match args[index].as_str() {
                "--process-request" => &mut input,
                "--process-result" => &mut output,
                other => return Err(format!("Unknown process-request argument {other}")),
            };
            if target.is_some() {
                return Err(format!("Duplicate {}", args[index]));
            }
            index += 1;
            let value = args
                .get(index)
                .filter(|value| !value.starts_with("--"))
                .ok_or_else(|| "Each process-request option requires a file path".to_string())?;
            *target = Some(PathBuf::from(value));
            index += 1;
        }
        let input = input.ok_or("--process-request REQUEST.json is required")?;
        let output = output.ok_or("--process-result RESULT.json is required")?;
        let request: ProcessRequest =
            serde_json::from_slice(&fs::read(&input).map_err(err)?).map_err(err)?;
        if output == input
            || output == request.path
            || fs::canonicalize(&output).ok().is_some_and(|resolved| {
                fs::canonicalize(&input).ok().as_ref() == Some(&resolved)
                    || fs::canonicalize(&request.path).ok().as_ref() == Some(&resolved)
            })
        {
            return Err("Result path must differ from request and source paths".into());
        }
        let result = process(&request, |message| eprintln!("PROCESS {message}"))?;
        write_json(&output, &result)?;
        Ok(())
    })())
}

pub fn process(
    request: &ProcessRequest,
    mut progress: impl FnMut(&str),
) -> Result<ProcessResult, String> {
    let started = Instant::now();
    let (valid_unix, domain) = validate_request(request)?;
    let source = fs::canonicalize(&request.path).map_err(err)?;
    progress("Verifying committed WRF source checksum and native time axis");
    let source_sha256 = sha256_file(&source)?;
    if source_sha256 != request.source_sha256.to_ascii_lowercase() {
        return Err("WRF source SHA-256 differs from the committed frame request".into());
    }
    let exact_time = inspect_source(&source, &domain, valid_unix, request.lead_seconds)?;
    let identity = FrameIdentity {
        case_id: request.case_id.clone(),
        source: "arwen".into(),
        model: format!("wrf-{domain}"),
        member: None,
        valid_unix,
        lead_seconds: exact_time.lead_seconds,
        source_sha256,
    };
    let viewer = if request.schema == VIEWER_REQUEST_SCHEMA {
        Some(ViewerProfile::new(&request.products)?)
    } else {
        None
    };
    let options = viewer
        .as_ref()
        .map(|profile| profile.options.clone())
        .unwrap_or_else(|| WrfProcessOptions {
            heavy_ecape: request.heavy_ecape,
            ..Default::default()
        });
    let result_schema = if viewer.is_some() {
        VIEWER_RESULT_SCHEMA
    } else {
        RESULT_SCHEMA
    };
    let id = if let Some(profile) = &viewer {
        digest(
            &serde_json::to_vec(&(
                "arwen-native-wrf-viewer-2d-v1",
                env!("GPUWM_BRIDGE_SOURCE_REV"),
                processor_fingerprint(),
                &identity,
                &options,
                &profile.products,
            ))
            .map_err(err)?,
        )
    } else {
        digest(
            &serde_json::to_vec(&(
                "arwen-native-wrf-full-v1",
                env!("GPUWM_BRIDGE_SOURCE_REV"),
                processor_fingerprint(),
                &identity,
                &options,
            ))
            .map_err(err)?,
        )
    };
    let requested_root = request.store_root.join(&id);
    fs::create_dir_all(&requested_root).map_err(err)?;
    let bundle_root = fs::canonicalize(&requested_root).map_err(err)?;
    let cache_receipt = bundle_root.join("processed-result.json");
    if cache_receipt.is_file() {
        let mut cached: ProcessResult =
            serde_json::from_slice(&fs::read(&cache_receipt).map_err(err)?).map_err(err)?;
        if cached.schema != result_schema
            || cached.frame.id != id
            || cached.frame.identity != identity
            || cached.domain != domain
            || cached.frame.store_root != bundle_root.join("rw-store")
        {
            return Err("Cached WRF conversion identity disagrees with its request".into());
        }
        verify_result(&cached)?;
        verify_viewer_result(&cached, viewer.as_ref())?;
        cached.cache_hit = true;
        progress(if viewer.is_some() {
            "Using the verified native 2-D viewer store"
        } else {
            "Using the verified native full WRF store"
        });
        return Ok(cached);
    }
    let store_root = bundle_root.join("rw-store");
    let case_sha256 = digest(&serde_json::to_vec(&(&request.case_id, &domain)).map_err(err)?);
    let target = LiveWrfTarget {
        case_sha256,
        storage_slot: 0,
        exact_time,
    };
    let task = spawn_process_live_path(source.clone(), store_root.clone(), options, target);
    let summary = loop {
        match task
            .rx
            .recv()
            .map_err(|error| format!("Native WRF processing worker stopped: {error}"))?
        {
            WrfProcessMessage::Progress(message) => progress(&message),
            WrfProcessMessage::Done(result) => break result?,
        }
    };
    if summary.hours_written != 1 || summary.files_seen != 1 {
        return Err("Native WRF processing must publish exactly one source/time".into());
    }
    if sha256_file(&source)? != identity.source_sha256 {
        return Err(
            "WRF source changed during processing; no companion frame was committed".into(),
        );
    }
    let directory = store_root.join(&summary.model).join(&summary.run);
    let hour_path = directory.join("f000.rws");
    let reader = HourReader::open(&hour_path).map_err(err)?;
    let meta = reader.meta();
    if meta.exact_time() != Some(exact_time) || meta.variables.is_empty() {
        return Err("Native WRF store returned an empty or differently timed frame".into());
    }
    if viewer.is_some()
        && meta
            .variables
            .iter()
            .any(|variable| variable.kind != "surface2d")
    {
        return Err("The 2-D viewer profile attempted to publish a non-plane variable".into());
    }
    let levels: BTreeSet<_> = meta
        .variables
        .iter()
        .flat_map(|variable| variable.levels_hpa.iter().copied())
        .collect();
    let frame = ProcessedFrame {
        schema: FRAME_SCHEMA.into(),
        id,
        identity,
        store_root,
        model_slug: summary.model,
        run_slug: summary.run,
        storage_slot: 0,
        hour_path: hour_path.clone(),
        grid_sha256: meta.grid_hash.clone(),
        shape: [meta.nx, meta.ny],
        variables: meta
            .variables
            .iter()
            .map(|variable| variable.name.clone())
            .collect(),
        levels_hpa: levels.into_iter().rev().collect(),
        rws_sha256: sha256_file(&hour_path)?,
        rws_bytes: fs::metadata(&hour_path).map_err(err)?.len(),
        processing_ms: started.elapsed().as_secs_f64() * 1000.,
    };
    let receipt_path = directory.join("companion-frame.json");
    let grid_path = directory.join("grid.rwg");
    let run_json_path = directory.join("run.json");
    // Validate the native hour/grid/run before publishing the companion receipt.
    verify_native(&frame, &grid_path, &run_json_path)?;
    write_json(&receipt_path, &frame)?;
    let files = [&hour_path, &grid_path, &run_json_path, &receipt_path]
        .into_iter()
        .map(|path| {
            Ok(BundleFile {
                path: path.clone(),
                sha256: sha256_file(path)?,
                bytes: fs::metadata(path).map_err(err)?.len(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let products = viewer
        .as_ref()
        .map(|profile| {
            profile.availability(
                &frame.store_root,
                &frame.model_slug,
                &frame.run_slug,
                frame.storage_slot,
            )
        })
        .transpose()?
        .unwrap_or_default();
    let members = if viewer.is_some() {
        members_for(&frame, &files)?
    } else {
        Vec::new()
    };
    let result = ProcessResult {
        schema: result_schema.into(),
        frame,
        domain,
        grid_path,
        run_json_path,
        receipt_path,
        cache_hit: false,
        notes: summary.notes,
        files,
        profile: viewer.as_ref().map(|_| VIEWER_PROFILE.into()),
        initialization_unix: viewer
            .as_ref()
            .map(|_| exact_time.origin_unix().expect("validated source time")),
        products,
        members,
    };
    write_json(&cache_receipt, &result)?;
    progress(&format!(
        "Native WRF store ready: {} fields, {} pressure levels, {:.1} MiB",
        result.frame.variables.len(),
        result.frame.levels_hpa.len(),
        result.frame.rws_bytes as f64 / 1048576.
    ));
    Ok(result)
}

fn validate_request(request: &ProcessRequest) -> Result<(i64, String), String> {
    match request.schema.as_str() {
        REQUEST_SCHEMA if request.profile.is_none() && request.products.is_empty() => {}
        VIEWER_REQUEST_SCHEMA
            if request.profile.as_deref() == Some(VIEWER_PROFILE) && !request.heavy_ecape => {}
        VIEWER_REQUEST_SCHEMA => {
            return Err(format!(
                "The v2 request requires profile {VIEWER_PROFILE:?} and no heavy ECAPE"
            ));
        }
        _ => {
            return Err(format!(
                "Expected an unchanged {REQUEST_SCHEMA} request or explicit {VIEWER_REQUEST_SCHEMA} viewer profile"
            ));
        }
    }
    if request.case_id.trim().is_empty() || request.case_id.len() > 512 {
        return Err("A nonempty case identity of at most 512 bytes is required".into());
    }
    if request.source_sha256.len() != 64
        || !request
            .source_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("source_sha256 must contain 64 hexadecimal digits".into());
    }
    let number = request
        .domain
        .strip_prefix('d')
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|number| *number > 0 && *number < 100)
        .ok_or("domain must identify one WRF nest, d01 through d99")?;
    let valid = chrono::DateTime::parse_from_rfc3339(&request.valid_utc)
        .map_err(|error| format!("valid_utc must be RFC3339: {error}"))?;
    if valid.timestamp_subsec_nanos() != 0 {
        return Err("WRF valid time must use whole seconds".into());
    }
    if request.store_root.as_os_str().is_empty() {
        return Err("A store root is required".into());
    }
    Ok((valid.timestamp(), format!("d{number:02}")))
}

fn inspect_source(
    path: &Path,
    domain: &str,
    valid_unix: i64,
    expected_lead: Option<u64>,
) -> Result<RwsExactTime, String> {
    let (axis, grid_id) = match wrf_core::WrfFile::open(path) {
        Ok(file) => (
            crate::local_import::wrf_source_times(&file, path)?,
            file.global_attr_i32("GRID_ID").ok().map(i64::from),
        ),
        Err(_) => {
            let file = netcrust::open(path).map_err(err)?;
            let grid_id = file
                .attribute("GRID_ID")
                .and_then(|attribute| attribute.as_f64())
                .filter(|value| value.is_finite() && value.fract() == 0.)
                .map(|value| value as i64);
            (
                crate::local_import::netcdf_source_times(&file, path).map_err(err)?,
                grid_id,
            )
        }
    };
    if axis.records.len() != 1 || axis.records[0].valid_unix != valid_unix {
        return Err(
            "Committed WRF frame must contain exactly the requested native valid time".into(),
        );
    }
    let grid_id = grid_id
        .or_else(|| {
            path.file_name()?
                .to_str()?
                .split('_')
                .find_map(|part| part.strip_prefix('d')?.parse::<i64>().ok())
        })
        .ok_or("WRF source provides neither GRID_ID nor a domain-bearing filename")?;
    if domain != format!("d{grid_id:02}") {
        return Err(format!(
            "WRF source domain d{grid_id:02} differs from requested {domain}"
        ));
    }
    let origin = axis
        .reference_unix
        .ok_or("WRF source has no authoritative initialization time")?;
    let lead_seconds = valid_unix
        .checked_sub(origin)
        .and_then(|lead| u64::try_from(lead).ok())
        .ok_or("WRF source valid time precedes initialization")?;
    if expected_lead.is_some_and(|lead| lead != lead_seconds) {
        return Err("WRF source lead time differs from requested lead_seconds".into());
    }
    Ok(RwsExactTime::new(lead_seconds, valid_unix))
}

fn verify_native(
    frame: &ProcessedFrame,
    grid_path: &Path,
    manifest_path: &Path,
) -> Result<(), String> {
    let directory = frame
        .store_root
        .join(&frame.model_slug)
        .join(&frame.run_slug);
    if frame.schema != FRAME_SCHEMA
        || frame.storage_slot != 0
        || frame.hour_path != directory.join("f000.rws")
        || grid_path != directory.join("grid.rwg")
        || manifest_path != directory.join("run.json")
    {
        return Err("Native WRF bundle paths or schema disagree".into());
    }
    let grid = GridFile::open(grid_path).map_err(err)?;
    let manifest = RwsRunManifest::load_for_run(manifest_path, &frame.model_slug, &frame.run_slug)
        .map_err(err)?;
    manifest
        .validate_grid(&grid.hash, grid.nx, grid.ny)
        .map_err(err)?;
    let reader = HourReader::open(&frame.hour_path).map_err(err)?;
    let meta = reader.meta();
    let exact = RwsExactTime::new(frame.identity.lead_seconds, frame.identity.valid_unix);
    if meta.model != frame.model_slug
        || meta.run != frame.run_slug
        || meta.forecast_hour != 0
        || meta.exact_time() != Some(exact)
        || manifest.exact_times().collect::<Vec<_>>() != vec![(0, exact)]
        || meta.grid_hash != grid.hash
        || frame.grid_sha256 != grid.hash
        || [meta.nx, meta.ny] != [grid.nx, grid.ny]
        || frame.shape != [grid.nx, grid.ny]
        || frame.variables
            != meta
                .variables
                .iter()
                .map(|variable| variable.name.clone())
                .collect::<Vec<_>>()
        || sha256_file(&frame.hour_path)? != frame.rws_sha256
    {
        return Err("Native WRF hour, grid, time, or source receipt verification failed".into());
    }
    Ok(())
}

pub fn verify_result(result: &ProcessResult) -> Result<(), String> {
    verify_native(&result.frame, &result.grid_path, &result.run_json_path)?;
    let expected = [
        &result.frame.hour_path,
        &result.grid_path,
        &result.run_json_path,
        &result.receipt_path,
    ];
    if result.receipt_path
        != result
            .frame
            .hour_path
            .with_file_name("companion-frame.json")
        || result.files.len() != expected.len()
    {
        return Err("Cached native WRF bundle file list disagrees".into());
    }
    for (file, path) in result.files.iter().zip(expected) {
        if &file.path != path
            || fs::metadata(path).map_err(err)?.len() != file.bytes
            || sha256_file(path)? != file.sha256
        {
            return Err("Cached native WRF bundle checksum changed".into());
        }
    }
    let committed: ProcessedFrame =
        serde_json::from_slice(&fs::read(&result.receipt_path).map_err(err)?).map_err(err)?;
    if committed.id != result.frame.id
        || committed.identity != result.frame.identity
        || committed.rws_sha256 != result.frame.rws_sha256
    {
        return Err("Committed WRF frame receipt changed".into());
    }
    Ok(())
}

fn members_for(frame: &ProcessedFrame, files: &[BundleFile]) -> Result<Vec<StoreMember>, String> {
    let identities = [
        ("fields", "rws_2d"),
        ("grid", "rwg"),
        ("run", "metadata"),
        ("frame", "metadata"),
    ];
    if files.len() != identities.len() {
        return Err("Viewer store needs exactly its four immutable native members".into());
    }
    files
        .iter()
        .zip(identities)
        .map(|(file, (key, kind))| {
            let relative = file.path.strip_prefix(&frame.store_root).map_err(err)?;
            let relative_path = relative
                .components()
                .map(|part| {
                    part.as_os_str()
                        .to_str()
                        .ok_or("Non-UTF-8 store member path")
                })
                .collect::<Result<Vec<_>, _>>()?
                .join("/");
            Ok(StoreMember {
                key: key.into(),
                relative_path,
                path: file.path.clone(),
                kind: kind.into(),
                bytes: file.bytes,
                sha256: file.sha256.clone(),
                grid_sha256: frame.grid_sha256.clone(),
            })
        })
        .collect()
}

fn verify_viewer_result(
    result: &ProcessResult,
    viewer: Option<&ViewerProfile>,
) -> Result<(), String> {
    let Some(viewer) = viewer else {
        if result.profile.is_some()
            || result.initialization_unix.is_some()
            || !result.products.is_empty()
            || !result.members.is_empty()
        {
            return Err("Full-science cache unexpectedly carries a viewer profile".into());
        }
        return Ok(());
    };
    let frame = &result.frame;
    let exact = RwsExactTime::new(frame.identity.lead_seconds, frame.identity.valid_unix);
    let reader = HourReader::open(&frame.hour_path).map_err(err)?;
    if result.profile.as_deref() != Some(VIEWER_PROFILE)
        || result.initialization_unix != exact.origin_unix()
        || reader
            .meta()
            .variables
            .iter()
            .any(|variable| variable.kind != "surface2d")
        || result.products
            != viewer.availability(
                &frame.store_root,
                &frame.model_slug,
                &frame.run_slug,
                frame.storage_slot,
            )?
        || result.members != members_for(frame, &result.files)?
    {
        return Err("Cached viewer profile, product availability, or member index changed".into());
    }
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(err)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(err)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn processor_fingerprint() -> String {
    let mut hash = Sha256::new();
    for source in [
        include_bytes!("process_request.rs").as_slice(),
        include_bytes!("wrf_process.rs").as_slice(),
        include_bytes!("wrf_volumes.rs").as_slice(),
        include_bytes!("local_import.rs").as_slice(),
        include_bytes!("postproc_severe.rs").as_slice(),
        include_bytes!("viewer_profile.rs").as_slice(),
        include_bytes!("wrf_chart_planes.rs").as_slice(),
        include_bytes!("../../../Cargo.lock").as_slice(),
    ] {
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source);
    }
    format!("{:x}", hash.finalize())
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(err)?;
    }
    rw_store::atomic::atomic_write_bytes(path, &serde_json::to_vec_pretty(value).map_err(err)?)
        .map_err(err)
}
fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn processing_cli_requires_both_paths_and_rejects_render_flags() {
        assert!(try_cli(&["--abi".into()]).is_none());
        for args in [
            vec!["--process-result"],
            vec!["--process-request", "a"],
            vec!["--process-request", "a", "--products", "all"],
            vec!["--process-request", "a", "--process-request", "b"],
        ] {
            assert!(
                try_cli(&args.into_iter().map(String::from).collect::<Vec<_>>())
                    .unwrap()
                    .is_err()
            );
        }
    }
}
