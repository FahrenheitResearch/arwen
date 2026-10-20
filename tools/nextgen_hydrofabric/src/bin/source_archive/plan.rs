//! Turn a selected original-source catalogue recipe into the existing archive
//! extractor's jobs. Planning reads metadata only; --check additionally streams
//! selected cached archives. --fetch acquires missing pinned HTTPS objects;
//! none of these modes extracts or decodes archive contents.
use nextgen_hydrofabric::Result;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path};
use std::time::Instant;

#[path = "fetch.rs"]
mod fetch;

#[derive(Deserialize)]
struct Catalog {
    schema: String,
    metadata: Metadata,
    objects: Vec<Object>,
    recipes: BTreeMap<String, Recipe>,
}
#[derive(Deserialize)]
struct Metadata {
    url: String,
    sha256: String,
    #[serde(rename = "Access_Constraints")]
    access: String,
    #[serde(rename = "Use_Constraints")]
    usage: String,
    source_attribution: String,
}
#[derive(Deserialize)]
struct Object {
    name: String,
    url: String,
    bytes: u64,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Recipe {
    source_row: String,
    region: String,
    scope: String,
    archives: Vec<Selection>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    object: String,
    members: Vec<String>,
    output: String,
    max_extracted_bytes: u64,
}
fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn relative(value: &str) -> bool {
    !value.is_empty()
        && !value.contains(['\\', ':'])
        && Path::new(value)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}
fn plan(raw: &[u8], recipe_id: &str, archives: &Path, output: &Path, budget: u64) -> Result<Value> {
    let c: Catalog = serde_json::from_slice(raw)?;
    if c.schema != "arwen.hydrofabric.public-source-catalog.v1" {
        return Err("unsupported public source catalogue schema".into());
    }
    // This catalogue is specifically the original federal NHDPlus source;
    // its permission cannot be inferred for a mirror or derived fabric.
    if c.metadata.access != "None."
        || c.metadata.usage != "None."
        || !sha(&c.metadata.sha256)
        || c.metadata.url.is_empty()
        || c.metadata.source_attribution.is_empty()
    {
        return Err("original source metadata permission/evidence incomplete".into());
    }
    if !archives.is_absolute()
        || !output.is_absolute()
        || archives.starts_with(output)
        || output.starts_with(archives)
        || archives
            .components()
            .chain(output.components())
            .any(|c| matches!(c, Component::ParentDir))
    {
        return Err("archive and extraction roots must be absolute, disjoint paths without parent traversal".into());
    }
    let recipe = c
        .recipes
        .get(recipe_id)
        .ok_or("unknown source acquisition recipe")?;
    if recipe.source_row.is_empty()
        || recipe.region.is_empty()
        || recipe.scope.is_empty()
        || recipe.archives.is_empty()
    {
        return Err("source acquisition recipe is incomplete".into());
    }
    let mut objects = BTreeMap::new();
    for object in &c.objects {
        if !relative(&object.name)
            || Path::new(&object.name).components().count() != 1
            || !sha(&object.sha256)
            || object.bytes == 0
            || !object.url.starts_with("https://")
            || objects.insert(&object.name, object).is_some()
        {
            return Err("invalid or duplicate pinned archive object".into());
        }
    }
    let mut selected = BTreeSet::new();
    let mut destinations: Vec<&Path> = Vec::new();
    let mut jobs = Vec::new();
    let mut downloads = Vec::new();
    let mut archive_bytes = 0u64;
    let mut extraction_budget = 0u64;
    for item in &recipe.archives {
        let object = objects
            .get(&item.object)
            .ok_or("recipe references an unpinned archive object")?;
        let path = Path::new(&item.output);
        if !selected.insert(&item.object)
            || !relative(&item.output)
            || destinations
                .iter()
                .any(|p| p.starts_with(path) || path.starts_with(p))
            || item.max_extracted_bytes == 0
            || item.members.is_empty()
            || item.members.iter().any(|m| !relative(m))
            || item.members.iter().collect::<BTreeSet<_>>().len() != item.members.len()
        {
            return Err("duplicate/unsafe archive selection, overlapping output or missing extraction bound".into());
        }
        destinations.push(path);
        archive_bytes = archive_bytes
            .checked_add(object.bytes)
            .ok_or("archive byte sum overflow")?;
        extraction_budget = extraction_budget
            .checked_add(item.max_extracted_bytes)
            .ok_or("extraction byte sum overflow")?;
        if archive_bytes > budget {
            return Err("selected archives exceed requested compressed-byte budget".into());
        }
        jobs.push(json!({"archive":archives.join(&object.name),"sha256":object.sha256,
            "members":item.members,"output":output.join(path),"max_extracted_bytes":item.max_extracted_bytes}));
        downloads.push(
            json!({"url":object.url,"bytes":object.bytes,"sha256":object.sha256,
            "path":archives.join(&object.name)}),
        );
    }
    Ok(
        json!({"schema":"arwen.hydrofabric.source-acquisition-plan.v1",
        "catalog_sha256":format!("{:x}",Sha256::digest(raw)),"recipe":recipe_id,
        "source_row":recipe.source_row,"region":recipe.region,"scope":recipe.scope,
        "source_metadata":{"url":c.metadata.url,"sha256":c.metadata.sha256,
            "access_constraints":c.metadata.access,"use_constraints":c.metadata.usage,
            "attribution":c.metadata.source_attribution},
        "compressed_bytes":archive_bytes,"max_extracted_bytes":extraction_budget,
        "downloads":downloads,"jobs":jobs,
        "limitations":"Exact selected source archives only; no download, file-presence verification, domain coverage, hydrologic parameters, repair or admitted runtime fabric."}),
    )
}
/// Inspect only selected files. A size failure never reads an oversized file;
/// hashing streams a bounded buffer per independent archive, never the dataset.
pub(super) fn check_archive(path: &Path, bytes: u64, expected: &str) -> Result<bool> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e.into()),
    };
    if !metadata.is_file() || metadata.len() != bytes {
        return Err(format!(
            "cached archive is not a regular file of pinned size: {}",
            path.display()
        )
        .into());
    }
    let mut input = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut count = 0u64;
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        count = count
            .checked_add(n as u64)
            .ok_or("archive byte count overflow")?;
        if count > bytes {
            return Err("cached archive grew beyond pinned size".into());
        }
        hash.update(&buffer[..n]);
    }
    if count != bytes || format!("{:x}", hash.finalize()) != expected {
        return Err(format!("cached archive SHA256/length mismatch: {}", path.display()).into());
    }
    Ok(true)
}
fn check_cache(result: &mut Value) -> Result<()> {
    let downloads = result["downloads"].as_array().ok_or("missing downloads")?;
    let present = downloads
        .par_iter()
        .map(|item| {
            check_archive(
                Path::new(item["path"].as_str().ok_or("missing archive path")?),
                item["bytes"].as_u64().ok_or("missing archive bytes")?,
                item["sha256"].as_str().ok_or("missing archive SHA256")?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut missing = Vec::new();
    let mut reusable_bytes = 0u64;
    let mut required_bytes = 0u64;
    let mut status = Vec::new();
    for (item, exists) in downloads.iter().zip(present) {
        let bytes = item["bytes"].as_u64().ok_or("missing archive bytes")?;
        if exists {
            reusable_bytes += bytes;
        } else {
            required_bytes += bytes;
            missing.push(item.clone());
        }
        status.push(json!({"path":item["path"], "bytes":bytes,
            "sha256":item["sha256"], "status":if exists { "verified" } else { "missing" }}));
    }
    result["cache"] = json!({"ready":missing.is_empty(),"reusable_bytes":reusable_bytes,
        "required_download_bytes":required_bytes,"required_downloads":missing,"archives":status,
        "rayon_workers":rayon::current_num_threads(),
        "parallelism":"one selected archive per Rayon job; sequential streaming SHA256, 65536-byte buffer per job"});
    result["limitations"] = json!("Selected archive bytes verified at check time only; extractor rechecks before decoding. No download, deletion, extraction, domain coverage, hydrologic parameters or admitted runtime fabric.");
    Ok(())
}
pub fn main_result() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(2).collect();
    if args.len() != 5 {
        return Err("usage: rw_hydrofabric_source_archive --plan|--check|--fetch CATALOG.json RECIPE ARCHIVE_ROOT EXTRACTION_ROOT MAX_COMPRESSED_BYTES".into());
    }
    let started = Instant::now();
    let raw = std::fs::read(&args[0])?;
    let mut result = plan(
        &raw,
        &args[1],
        Path::new(&args[2]),
        Path::new(&args[3]),
        args[4].parse()?,
    )?;
    let mode = std::env::args().nth(1).unwrap_or_default();
    if mode == "--check" || mode == "--fetch" {
        check_cache(&mut result)?;
    }
    if mode == "--fetch" {
        result["acquisition"] = fetch::missing(&result)?;
        check_cache(&mut result)?;
        result["limitations"] = json!("Selected original archives acquired and verified only; extractor rechecks before decoding. No extraction, domain coverage, hydrologic parameters or admitted runtime fabric.");
    }
    result["wall_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    result["parallelism"] = if result.get("acquisition").is_some() {
        result["acquisition"]["parallelism"].clone()
    } else if result.get("cache").is_some() {
        result["cache"]["parallelism"].clone()
    } else {
        json!("single metadata planner; extraction retains one independent archive per Rayon job")
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Value {
        json!({"schema":"arwen.hydrofabric.public-source-catalog.v1",
            "metadata":{"url":"https://example.org/metadata","sha256":"a".repeat(64),
                "Access_Constraints":"None.","Use_Constraints":"None.","source_attribution":"agency"},
            "objects":[{"name":"one.7z","url":"https://example.org/one","bytes":10,"sha256":"b".repeat(64)},
                       {"name":"two.7z","url":"https://example.org/two","bytes":20,"sha256":"c".repeat(64)}],
            "recipes":{"region-one":{"source_row":"source","region":"one","scope":"source only",
                "archives":[{"object":"one.7z","members":["folder/table.dbf"],"output":"attributes","max_extracted_bytes":100}]}}})
    }
    fn run(v: Value) -> Result<Value> {
        plan(
            &serde_json::to_vec(&v)?,
            "region-one",
            Path::new("/archives"),
            Path::new("/extracted"),
            10,
        )
    }
    #[test]
    fn selects_only_requested_region_and_keeps_hashes_and_credit() {
        let result = run(fixture()).unwrap();
        assert_eq!(result["compressed_bytes"], 10);
        assert_eq!(result["jobs"].as_array().unwrap().len(), 1);
        assert_eq!(result["jobs"][0]["archive"], "/archives/one.7z");
        assert_eq!(result["jobs"][0]["sha256"], "b".repeat(64));
        assert_eq!(result["source_metadata"]["attribution"], "agency");
    }
    #[test]
    fn refuses_unpinned_or_unbounded_or_unsafe_selection() {
        for field in ["object", "output", "members", "max_extracted_bytes"] {
            let mut v = fixture();
            v["recipes"]["region-one"]["archives"][0][field] = match field {
                "object" => json!("unknown"),
                "output" => json!("../outside"),
                "members" => json!(["../outside"]),
                _ => json!(0),
            };
            assert!(run(v).is_err(), "{field}");
        }
        let mut v = fixture();
        v["objects"][0]["bytes"] = json!(11);
        assert!(run(v)
            .unwrap_err()
            .to_string()
            .contains("compressed-byte budget"));
        let mut v = fixture();
        v["metadata"]["Use_Constraints"] = json!("restricted");
        assert!(run(v).is_err());
        let mut v = fixture();
        v["objects"][0]["sha256"] = json!("unknown");
        assert!(run(v).is_err());
    }
    #[test]
    fn refuses_overlapping_outputs_and_duplicate_members() {
        let mut v = fixture();
        let mut item = v["recipes"]["region-one"]["archives"][0].clone();
        item["object"] = json!("two.7z");
        item["output"] = json!("attributes/nested");
        v["recipes"]["region-one"]["archives"]
            .as_array_mut()
            .unwrap()
            .push(item);
        assert!(run(v).unwrap_err().to_string().contains("overlapping"));
        let mut v = fixture();
        v["recipes"]["region-one"]["archives"][0]["members"] = json!(["x", "x"]);
        assert!(run(v).is_err());
    }

    #[test]
    fn distinguishes_missing_verified_corrupt_and_nonregular_cache() {
        let root = std::path::PathBuf::from(
            std::env::var_os("TMPDIR").expect("set scratch TMPDIR"),
        )
        .join(format!(
            "source-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("archive");
        let hash = format!("{:x}", Sha256::digest(b"original"));
        assert!(!check_archive(&path, 8, &hash).unwrap());
        std::fs::write(&path, b"original").unwrap();
        assert!(check_archive(&path, 8, &hash).unwrap());
        std::fs::write(&path, b"altered!").unwrap();
        assert!(check_archive(&path, 8, &hash)
            .unwrap_err()
            .to_string()
            .contains("SHA256"));
        assert!(check_archive(&path, 7, &hash)
            .unwrap_err()
            .to_string()
            .contains("pinned size"));
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(check_archive(&path, 8, &hash).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
