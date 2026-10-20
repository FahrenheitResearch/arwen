//! Startup-only reader for the existing prepared hydrofabric manifest.
//! Driver owns allocation/binding; this module never initializes model state.
use crate::Result;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::{c_char, CStr},
    fs,
    path::{Component, Path, PathBuf},
};

pub const SCHEMA: &str = "arwen.nextgen.hydrofabric.v1";

#[derive(Debug, Deserialize)]
pub struct Field {
    pub path: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub sha256: String,
}

impl Field {
    pub fn validate(&self) -> Result<()> {
        relative(&self.path)?;
        digest_text(&self.sha256)?;
        let width = match self.dtype.as_str() {
            "<f4" | "<i4" | "<u4" => 4usize,
            "<f8" | "<i8" | "<u8" => 8usize,
            _ => return Err(format!("unsupported hydrofabric dtype {}", self.dtype).into()),
        };
        let elements = self
            .shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or("hydrofabric shape product overflow")?;
        if elements.checked_mul(width) != Some(self.bytes) {
            return Err("hydrofabric descriptor shape/dtype/byte count mismatch".into());
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct Input {
    pub path: String,
    pub sha256: String,
    pub data_license: String,
}

#[derive(Debug, Deserialize)]
pub struct Auxiliary {
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
pub struct Document {
    pub schema: String,
    pub request_sha256: String,
    pub input: Input,
    pub node_count: usize,
    pub catchment_count: usize,
    pub level_count: usize,
    pub node_order: String,
    pub upstream_order: String,
    pub numerics: String,
    pub files: BTreeMap<String, Field>,
    pub auxiliary: BTreeMap<String, Auxiliary>,
    /// Grid, routing, reservoirs, parent identity and other existing metadata.
    #[serde(flatten)]
    pub metadata: BTreeMap<String, Value>,
}

fn digest_text(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err("hydrofabric SHA256 must be 64 lowercase hexadecimal characters".into());
    }
    Ok(())
}

fn relative(value: &str) -> Result<()> {
    if value.is_empty()
        || Path::new(value)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err("hydrofabric payload path must be relative without parent components".into());
    }
    Ok(())
}

pub struct PreparedManifest {
    pub document: Document,
    root: PathBuf,
}

impl PreparedManifest {
    /// Parse/validate descriptors once at setup. Source paths are provenance;
    /// loading a prepared artifact never reopens its original GPKG/capture.
    pub fn open(path: &Path) -> Result<Self> {
        let absolute = path.canonicalize()?;
        let document: Document = serde_json::from_slice(&fs::read(&absolute)?)?;
        if document.schema != SCHEMA {
            return Err("unsupported prepared hydrofabric manifest schema".into());
        }
        digest_text(&document.request_sha256)?;
        digest_text(&document.input.sha256)?;
        if document.input.data_license.trim().is_empty() {
            return Err("prepared hydrofabric data_license is empty".into());
        }
        for field in document.files.values() {
            field.validate()?;
        }
        for (name, auxiliary) in &document.auxiliary {
            relative(name)?;
            digest_text(&auxiliary.sha256)?;
        }
        Ok(Self {
            root: absolute.parent().ok_or("manifest has no directory")?.into(),
            document,
        })
    }

    fn read(&self, relative_path: &str, expected: &str, length: Option<usize>) -> Result<Vec<u8>> {
        relative(relative_path)?;
        let path = self.root.join(relative_path).canonicalize()?;
        if !path.starts_with(&self.root) {
            return Err("hydrofabric payload resolves outside prepared directory".into());
        }
        let bytes = fs::read(&path)?;
        if length.is_some_and(|n| n != bytes.len()) {
            return Err("prepared hydrofabric payload length mismatch".into());
        }
        if format!("{:x}", Sha256::digest(&bytes)) != expected {
            return Err("prepared hydrofabric payload SHA256 mismatch".into());
        }
        Ok(bytes)
    }

    /// Copies only after descriptor, path, length and complete SHA validation.
    pub fn read_field_into(&self, name: &str, destination: &mut [u8]) -> Result<()> {
        let field = self
            .document
            .files
            .get(name)
            .ok_or("hydrofabric field absent from manifest")?;
        if destination.len() != field.bytes {
            return Err("hydrofabric destination length differs from descriptor".into());
        }
        let bytes = self.read(&field.path, &field.sha256, Some(field.bytes))?;
        destination.copy_from_slice(&bytes);
        Ok(())
    }

    /// Hash-check network.json/other declared metadata before the driver uses it.
    pub fn read_auxiliary(&self, name: &str) -> Result<Vec<u8>> {
        let field = self
            .document
            .auxiliary
            .get(name)
            .ok_or("hydrofabric auxiliary absent from manifest")?;
        self.read(name, &field.sha256, None)
    }
}

/// Additive ABI1 helper for bindings that allocate their own pinned transport.
/// Status: 0 success, 1 invalid pointer/string, 2 manifest/payload error, 3 panic.
/// Destination is unchanged on validation failure. Calls are startup-only.
/// # Safety
/// Both strings are valid NUL-terminated UTF-8, destination is writable for
/// length bytes and does not alias either string or the manifest file data.
#[no_mangle]
pub unsafe extern "C" fn arwen_hydrofabric_manifest_read_into(
    manifest_path: *const c_char,
    field_name: *const c_char,
    destination: *mut u8,
    length: usize,
) -> i32 {
    if manifest_path.is_null() || field_name.is_null() || (destination.is_null() && length != 0) {
        return 1;
    }
    let (Ok(path), Ok(name)) = (
        CStr::from_ptr(manifest_path).to_str(),
        CStr::from_ptr(field_name).to_str(),
    ) else {
        return 1;
    };
    match std::panic::catch_unwind(|| -> Result<()> {
        let prepared = PreparedManifest::open(Path::new(path))?;
        let target = if length == 0 {
            &mut []
        } else {
            std::slice::from_raw_parts_mut(destination, length)
        };
        prepared.read_field_into(name, target)
    }) {
        Ok(Ok(())) => 0,
        Ok(Err(_)) => 2,
        Err(_) => 3,
    }
}
