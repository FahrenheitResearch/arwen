use crate::publication::{atomic_write_bytes, sha256_hex};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

const BINCODE_CACHE_SCHEMA_VERSION: u32 = 2;
const LEGACY_BINCODE_CACHE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
struct VersionedCachePayload {
    schema_version: u32,
    payload_bincode_len: usize,
    payload_sha256: String,
    payload_bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
struct LegacyVersionedCachePayload<T> {
    schema_version: u32,
    payload: T,
}

pub fn default_proof_cache_dir(out_dir: &Path) -> PathBuf {
    out_dir.join("cache")
}

pub fn load_bincode<T: DeserializeOwned>(
    path: &Path,
) -> Result<Option<T>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    if let Ok(wrapper) = bincode::deserialize::<VersionedCachePayload>(&bytes) {
        if wrapper.schema_version == BINCODE_CACHE_SCHEMA_VERSION {
            let payload_sha256 = sha256_hex(&wrapper.payload_bytes);
            if wrapper.payload_bincode_len != wrapper.payload_bytes.len()
                || wrapper.payload_sha256 != payload_sha256
            {
                quarantine_cache_file(path, "payload_attestation_mismatch");
                return Ok(None);
            }
            match bincode::deserialize::<T>(&wrapper.payload_bytes) {
                Ok(value) => return Ok(Some(value)),
                Err(_) => {
                    quarantine_cache_file(path, "payload_decode_error");
                    return Ok(None);
                }
            }
        }
        if wrapper.schema_version == LEGACY_BINCODE_CACHE_SCHEMA_VERSION {
            quarantine_cache_file(path, "schema_mismatch");
            return Ok(None);
        }
        quarantine_cache_file(path, "schema_mismatch");
        return Ok(None);
    }
    if let Ok(wrapper) = bincode::deserialize::<LegacyVersionedCachePayload<T>>(&bytes) {
        if wrapper.schema_version == LEGACY_BINCODE_CACHE_SCHEMA_VERSION {
            return Ok(Some(wrapper.payload));
        }
        quarantine_cache_file(path, "schema_mismatch");
        return Ok(None);
    }
    if let Ok(value) = bincode::deserialize::<T>(&bytes) {
        return Ok(Some(value));
    }
    quarantine_cache_file(path, "decode_error");
    Ok(None)
}

pub fn store_bincode<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload_bytes = bincode::serialize(value)?;
    let bytes = bincode::serialize(&VersionedCachePayload {
        schema_version: BINCODE_CACHE_SCHEMA_VERSION,
        payload_bincode_len: payload_bytes.len(),
        payload_sha256: sha256_hex(&payload_bytes),
        payload_bytes,
    })?;
    atomic_write_bytes(path, &bytes)?;
    Ok(())
}

pub fn ensure_dir(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
    struct Fixture {
        name: String,
        value: u16,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum LegacyProjectionFixture {
        Geographic,
        LambertConformal {
            standard_parallel_1_deg: f64,
            standard_parallel_2_deg: f64,
            central_meridian_deg: f64,
        },
    }

    #[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
    struct LegacyProjectionPayload {
        name: String,
        projection: LegacyProjectionFixture,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum ProjectionFixture {
        Geographic,
        LambertConformal {
            standard_parallel_1_deg: f64,
            standard_parallel_2_deg: f64,
            central_meridian_deg: f64,
        },
    }

    #[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
    struct ProjectionPayload {
        name: String,
        projection: ProjectionFixture,
    }

    #[test]
    fn bincode_round_trip_works() {
        let root =
            std::env::temp_dir().join(format!("rustwx_products_cache_{}", std::process::id()));
        let path = root.join("fixture.bin");
        let fixture = Fixture {
            name: "demo".into(),
            value: 7,
        };

        store_bincode(&path, &fixture).unwrap();
        let loaded = load_bincode::<Fixture>(&path).unwrap().unwrap();
        assert_eq!(loaded, fixture);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn versioned_bincode_payload_records_attestation_metadata() {
        let root = std::env::temp_dir().join(format!(
            "rustwx_products_cache_attestation_{}",
            std::process::id()
        ));
        let path = root.join("fixture.bin");
        let fixture = Fixture {
            name: "attested".into(),
            value: 11,
        };

        store_bincode(&path, &fixture).unwrap();
        let bytes = fs::read(&path).unwrap();
        let wrapper: VersionedCachePayload = bincode::deserialize(&bytes).unwrap();
        let payload_bytes = bincode::serialize(&fixture).unwrap();
        assert_eq!(wrapper.schema_version, BINCODE_CACHE_SCHEMA_VERSION);
        assert_eq!(wrapper.payload_bincode_len, payload_bytes.len());
        assert_eq!(wrapper.payload_sha256, sha256_hex(&payload_bytes));
        assert_eq!(wrapper.payload_bytes, payload_bytes);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_bincode_payload_still_loads() {
        let root = std::env::temp_dir().join(format!(
            "rustwx_products_cache_legacy_{}",
            std::process::id()
        ));
        let path = root.join("fixture.bin");
        let fixture = Fixture {
            name: "legacy".into(),
            value: 9,
        };

        fs::create_dir_all(&root).unwrap();
        fs::write(&path, bincode::serialize(&fixture).unwrap()).unwrap();
        let loaded = load_bincode::<Fixture>(&path).unwrap().unwrap();
        assert_eq!(loaded, fixture);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn tampered_versioned_payload_is_quarantined_and_treated_as_cache_miss() {
        let root = std::env::temp_dir().join(format!(
            "rustwx_products_cache_tampered_{}",
            std::process::id()
        ));
        let path = root.join("fixture.bin");
        let fixture = Fixture {
            name: "tampered".into(),
            value: 5,
        };

        fs::create_dir_all(&root).unwrap();
        let payload_bytes = bincode::serialize(&fixture).unwrap();
        let bytes = bincode::serialize(&VersionedCachePayload {
            schema_version: BINCODE_CACHE_SCHEMA_VERSION,
            payload_bincode_len: payload_bytes.len(),
            payload_sha256: "deadbeef".into(),
            payload_bytes,
        })
        .unwrap();
        fs::write(&path, bytes).unwrap();

        let loaded = load_bincode::<Fixture>(&path).unwrap();
        assert!(loaded.is_none());
        assert!(!path.exists());
        let quarantined = fs::read_dir(&root)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            quarantined
                .iter()
                .any(|entry| entry.file_name().to_string_lossy().contains("corrupt"))
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn corrupt_bincode_payload_is_quarantined_and_treated_as_cache_miss() {
        let root = std::env::temp_dir().join(format!(
            "rustwx_products_cache_corrupt_{}",
            std::process::id()
        ));
        let path = root.join("fixture.bin");

        fs::create_dir_all(&root).unwrap();
        fs::write(&path, b"not-bincode").unwrap();
        let loaded = load_bincode::<Fixture>(&path).unwrap();
        assert!(loaded.is_none());
        assert!(!path.exists());
        let quarantined = fs::read_dir(&root)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            quarantined
                .iter()
                .any(|entry| entry.file_name().to_string_lossy().contains("corrupt"))
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn incompatible_inner_payload_is_quarantined_and_treated_as_cache_miss() {
        let root = std::env::temp_dir().join(format!(
            "rustwx_products_cache_incompatible_{}",
            std::process::id()
        ));
        let path = root.join("fixture.bin");
        let legacy = LegacyProjectionPayload {
            name: "legacy".into(),
            projection: LegacyProjectionFixture::LambertConformal {
                standard_parallel_1_deg: 38.5,
                standard_parallel_2_deg: 38.5,
                central_meridian_deg: -97.5,
            },
        };

        fs::create_dir_all(&root).unwrap();
        let payload_bytes = bincode::serialize(&legacy).unwrap();
        let bytes = bincode::serialize(&VersionedCachePayload {
            schema_version: BINCODE_CACHE_SCHEMA_VERSION,
            payload_bincode_len: payload_bytes.len(),
            payload_sha256: sha256_hex(&payload_bytes),
            payload_bytes,
        })
        .unwrap();
        fs::write(&path, bytes).unwrap();

        let loaded = load_bincode::<ProjectionPayload>(&path).unwrap();
        assert!(loaded.is_none());
        assert!(!path.exists());
        let quarantined = fs::read_dir(&root)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            quarantined
                .iter()
                .any(|entry| entry.file_name().to_string_lossy().contains("corrupt"))
        );

        let _ = fs::remove_dir_all(root);
    }
}

fn quarantine_cache_file(path: &Path, reason: &str) {
    if !path.exists() {
        return;
    }
    let quarantine_path = quarantine_path_for(path, reason);
    if let Some(parent) = quarantine_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if fs::rename(path, &quarantine_path).is_err() {
        let _ = fs::remove_file(path);
    }
}

fn quarantine_path_for(path: &Path, reason: &str) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache");
    path.with_file_name(format!(
        "{file_name}.corrupt-{reason}-{}-{}",
        process::id(),
        unique_suffix()
    ))
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}
