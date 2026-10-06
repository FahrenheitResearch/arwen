//! Exact domain/time and immutable bundle contract around the original WRF engine.
mod stored_plane_fixture;
use rw_wrfbatch::process_request::{ProcessRequest, REQUEST_SCHEMA, process, sha256_file};
use std::path::PathBuf;

struct Scratch(PathBuf);
// Tests in this file run on parallel threads of one process; two of them read the same clock
// tick on the 2.8.6 public Windows runner, shared one directory, and one's Drop removed it under
// the other. The counter makes every scratch root distinct.
static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
impl Scratch {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "arwen-full-wrf-request-{}-{}-{}",
            std::process::id(),
            NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn request(scratch: &Scratch) -> ProcessRequest {
    let original = stored_plane_fixture::write_rain_frame(&scratch.0, 900, 2.);
    let sha = sha256_file(&original).unwrap();
    let path = scratch.0.join(format!("{sha}.wrf"));
    std::fs::rename(original, &path).unwrap();
    ProcessRequest {
        schema: REQUEST_SCHEMA.into(),
        path,
        source_sha256: sha,
        case_id: "subhour-case".into(),
        domain: "d01".into(),
        valid_utc: "2026-08-19T00:15:00Z".into(),
        store_root: scratch.0.join("stores"),
        lead_seconds: Some(900),
        heavy_ecape: false,
        profile: None,
        products: Vec::new(),
    }
}
#[test]
fn immutable_subhour_request_retains_native_planes_profiles_and_verified_cache() {
    let scratch = Scratch::new();
    let request = request(&scratch);
    let result = process(&request, |_| {}).unwrap();
    assert!(!result.cache_hit);
    assert_eq!(result.frame.storage_slot, 0);
    assert_eq!(result.frame.identity.lead_seconds, 900);
    assert_eq!(result.frame.identity.model, "wrf-d01");
    assert_eq!(result.files.len(), 4);
    assert_eq!(result.frame.identity.source_sha256, request.source_sha256);
    assert!(
        result
            .frame
            .variables
            .contains(&stored_plane_fixture::USER_PLANE_STORE_NAME.into())
    );
    for field in [
        "temperature_iso",
        "dewpoint_iso",
        "u_iso",
        "v_iso",
        "height_iso",
    ] {
        assert!(
            result.frame.variables.contains(&field.into()),
            "missing {field}"
        );
    }
    let reader = rw_store::reader::HourReader::open(&result.frame.hour_path).unwrap();
    let values = reader
        .read_full_2d(stored_plane_fixture::USER_PLANE_STORE_NAME)
        .unwrap();
    assert_eq!(values[0], stored_plane_fixture::user_plane_value(0, 0));
    drop(reader);
    let cached = process(&request, |_| {}).unwrap();
    assert!(cached.cache_hit);
    assert_eq!(result.frame.id, cached.frame.id);
    let mut changed = request.clone();
    changed.case_id = "other-case".into();
    let other = process(&changed, |_| {}).unwrap();
    assert_ne!(result.frame.store_root, other.frame.store_root);
    let grid = std::fs::read(&result.grid_path).unwrap();
    let mut broken = grid.clone();
    broken[0] ^= 1;
    std::fs::write(&result.grid_path, broken).unwrap();
    assert!(process(&request, |_| {}).is_err());
    assert_eq!(sha256_file(&request.path).unwrap(), request.source_sha256);
}
#[test]
fn wrong_source_hash_domain_time_and_lead_are_refused_before_store_publication() {
    let scratch = Scratch::new();
    let request = request(&scratch);
    for wrong in 0..4 {
        let mut changed = request.clone();
        match wrong {
            0 => changed.source_sha256 = "0".repeat(64),
            1 => changed.domain = "d02".into(),
            2 => changed.valid_utc = "2026-08-19T00:16:00Z".into(),
            _ => changed.lead_seconds = Some(901),
        }
        assert!(process(&changed, |_| {}).is_err());
        assert!(!request.store_root.exists());
    }
}

#[test]
fn viewer_profile_preserves_exact_time_and_selected_science_without_volume_storage() {
    use rw_wrfbatch::process_request::{VIEWER_REQUEST_SCHEMA, VIEWER_RESULT_SCHEMA};
    let scratch = Scratch::new();
    let mut request = request(&scratch);
    let full = process(&request, |_| {}).unwrap();
    request.schema = VIEWER_REQUEST_SCHEMA.into();
    request.profile = Some("viewer-2d-v1".into());
    let viewer = process(&request, |_| {}).unwrap();
    assert_eq!(viewer.schema, VIEWER_RESULT_SCHEMA);
    assert_eq!(viewer.profile.as_deref(), Some("viewer-2d-v1"));
    assert_eq!(viewer.frame.identity, full.frame.identity);
    assert_ne!(viewer.frame.id, full.frame.id);
    assert_eq!(
        viewer.initialization_unix,
        Some(viewer.frame.identity.valid_unix - 900)
    );
    assert_eq!(viewer.products.len(), 20);
    assert_eq!(viewer.members.len(), 4);
    assert_eq!(viewer.members[0].key, "fields");
    assert_eq!(viewer.members[0].kind, "rws_2d");
    assert!(viewer.frame.levels_hpa.is_empty());
    assert!(
        !viewer
            .frame
            .variables
            .contains(&stored_plane_fixture::USER_PLANE_STORE_NAME.into())
    );
    let full_reader = rw_store::reader::HourReader::open(&full.frame.hour_path).unwrap();
    let reader = rw_store::reader::HourReader::open(&viewer.frame.hour_path).unwrap();
    assert!(
        reader
            .meta()
            .variables
            .iter()
            .all(|field| field.kind == "surface2d")
    );
    assert!(reader.meta().variables.len() < full_reader.meta().variables.len());
    for field in &reader.meta().variables {
        assert!(
            full_reader.variable(&field.name).is_some(),
            "extra viewer field {}",
            field.name
        );
        let values = reader.read_full_2d(&field.name).unwrap();
        let expected = full_reader.read_full_2d(&field.name).unwrap();
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{} differs",
            field.name
        );
    }
    assert!(process(&request, |_| {}).unwrap().cache_hit);
    request.products = vec!["2m_temperature".into()];
    let single = process(&request, |_| {}).unwrap();
    assert_eq!(single.products.len(), 1);
    assert!(single.products[0].available);
    assert_eq!(single.frame.variables.len(), 2); // requested temperature plus source terrain
    assert_ne!(single.frame.id, viewer.frame.id);
    assert_eq!(sha256_file(&request.path).unwrap(), request.source_sha256);
}

/// The Python door takes `var:<stored 2-D variable>` for the compact
/// viewer, and the viewer refused every one of them with `Unknown or
/// non-2-D viewer product`, so no stored field could be asked for by name.
#[test]
fn viewer_stored_field_selection_is_drawn_from_the_stored_plane() {
    use rw_wrfbatch::process_request::VIEWER_REQUEST_SCHEMA;
    let scratch = Scratch::new();
    let mut request = request(&scratch);
    let full = process(&request, |_| {}).unwrap();
    request.schema = VIEWER_REQUEST_SCHEMA.into();
    request.profile = Some("viewer-2d-v1".into());
    // A core plane a named product also draws, a chart-level plane, a raw
    // WRF field, a field a user added to their own Registry, and a name
    // this frame does not store.
    let stored = [
        "temperature_2m",
        "geopotential_height_850hpa",
        "wrf_tsk",
        stored_plane_fixture::USER_PLANE_STORE_NAME,
    ];
    request.products = stored
        .iter()
        .map(|name| format!("var:{name}"))
        .chain(["var:no_such_field".to_string()])
        .collect();
    let viewer = process(&request, |_| {}).unwrap();
    let status = |slug: &str| {
        viewer
            .products
            .iter()
            .find(|row| row.slug == slug)
            .unwrap_or_else(|| panic!("no status for {slug}: {:?}", viewer.products))
    };
    let full_reader = rw_store::reader::HourReader::open(&full.frame.hour_path).unwrap();
    let reader = rw_store::reader::HourReader::open(&viewer.frame.hour_path).unwrap();
    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for name in stored {
        let row = status(&format!("var:{name}"));
        assert!(row.available, "{row:?}");
        assert_eq!(row.source_fields, [name]);
        assert_eq!(
            bits(reader.read_full_2d(name).unwrap()),
            bits(full_reader.read_full_2d(name).unwrap()),
            "{name} differs from the full-science store"
        );
    }
    let missing = status("var:no_such_field");
    assert!(!missing.available);
    assert_eq!(
        missing.missing_reasons,
        ["This frame stores no 2-D field named no_such_field"]
    );
    // Only the named fields and the terrain plane are processed.
    let mut variables = viewer.frame.variables.clone();
    variables.sort();
    let mut expected: Vec<String> = stored
        .iter()
        .map(|name| name.to_string())
        .chain(["orography".to_string()])
        .collect();
    expected.sort();
    assert_eq!(variables, expected);
    assert!(viewer.frame.levels_hpa.is_empty());
    assert!(process(&request, |_| {}).unwrap().cache_hit);
}

#[test]
fn viewer_request_requires_explicit_version_and_does_not_override_the_full_profile() {
    use rw_wrfbatch::process_request::VIEWER_REQUEST_SCHEMA;
    let scratch = Scratch::new();
    let mut request = request(&scratch);
    request.profile = Some("viewer-2d-v1".into());
    assert!(process(&request, |_| {}).is_err());
    request.schema = VIEWER_REQUEST_SCHEMA.into();
    request.heavy_ecape = true;
    assert!(process(&request, |_| {}).is_err());
    request.heavy_ecape = false;
    request.products = vec!["unknown-product".into()];
    assert!(process(&request, |_| {}).is_err());
    assert!(!request.store_root.exists());
}
