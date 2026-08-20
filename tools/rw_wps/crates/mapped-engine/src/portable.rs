//! The machine-INDEPENDENT form of a canonical frame header.
//!
//! Port of `gpuwm.source_frame.portable_frame_header`; the two must agree
//! byte for byte, because the parity battery compares the documents both
//! engines publish and the crate goldens assert the digest this produces.
//!
//! A canonical frame header is an exact identity of a decode, which is
//! what makes it worth digesting -- and two members of it are identities
//! of the BOX instead:
//!
//! * every field descriptor quotes the absolute paths its arrays were read
//!   from, so the digest moves with a checkout, a staging tree or a drive
//!   letter;
//! * a field produced through a TRANSCENDENTAL takes its last bits from
//!   the box's libm.  Measured 2026-08-20, same bytes on the Windows
//!   desktop (UCRT) and weather-node-1 (glibc): the two `exp`-based
//!   humidity derivations differ by at most 3 ULP (4.1e-16 relative) on
//!   the netCDF case, and the `sin`/`cos` grid-relative wind rotation
//!   moves all four wind components on the Lambert case.  Every other
//!   array -- integer unpack plus IEEE add, multiply and divide -- is
//!   identical to the byte.
//!
//! So exactly two normalizations are applied, both declared, neither
//! per-source (the libm-dependent set is read from the mapping's own
//! declarations by [`libm_dependent_fields`], so a new model stays table
//! work):
//!
//! 1. each input path is reduced to its file name;
//! 2. each libm-dependent field's `data_reference` -- an array digest --
//!    becomes `libm:<canonical_name>`.
//!
//! What survives is the whole gate: grid, vertical coordinates, times,
//! policies, units, shapes, dtypes, the field roster BY NAME, and the
//! exact array digest of every other field.  The elided arrays are not
//! left uncompared; they are compared by value under a declared tolerance
//! instead of by digest (see `tests/goldens.rs`).

use std::collections::BTreeSet;

use serde_json::{Map, Value};

/// The declared normalization, versioned: a recorded portable digest is
/// only comparable against the same rule.
pub const PORTABLE_HEADER_RULE: &str = "gpuwm-portable-frame-header-v1";

/// The fields a mapping produces through a transcendental.
///
/// Port of `gpuwm.mapped_source.libm_dependent_fields`.  Two productions
/// in this engine call `exp`/`sin`/`cos`, and a transcendental's last bit
/// is the box's libm, not the decode's answer: a field with a declared
/// `derivation`, and both wind pairs when the declaration says the source
/// publishes grid-relative components this engine rotates.  Everything
/// else is integer unpack plus IEEE add/multiply/divide, which is
/// bit-reproducible on any conforming machine.
///
/// Read from the mapping's own declarations, so a new model is table
/// work: nothing here names a source.
pub fn libm_dependent_fields(mapping: &crate::model::Mapping) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let Ok(fields) = mapping.fields() else {
        return names;
    };
    let declared: BTreeSet<String> = fields.iter().map(|field| field.name.clone()).collect();
    for field in &fields {
        if field.derivation().is_some() {
            names.insert(field.name.clone());
        }
    }
    if mapping
        .grid_declaration()
        .map(|declaration| declaration.rotates_winds())
        .unwrap_or(false)
    {
        for (u_name, v_name) in crate::model::ROTATED_WIND_PAIRS {
            for name in [u_name, v_name] {
                if declared.contains(name) {
                    names.insert(name.to_owned());
                }
            }
        }
    }
    names
}

/// The file name of a path, on either platform's separator.
fn file_name(spelling: &str) -> &str {
    spelling
        .rsplit(|character| character == '/' || character == '\\')
        .next()
        .unwrap_or(spelling)
}

fn mask_input_paths(value: &Value, spellings: &[(String, String)]) -> Value {
    match value {
        Value::String(text) => {
            let mut masked = text.clone();
            for (spelling, name) in spellings {
                masked = masked.replace(spelling.as_str(), name.as_str());
            }
            Value::String(masked)
        }
        Value::Array(items) => {
            Value::Array(items.iter().map(|item| mask_input_paths(item, spellings)).collect())
        }
        Value::Object(members) => Value::Object(
            members
                .iter()
                .map(|(key, item)| (key.clone(), mask_input_paths(item, spellings)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The header with the two declared normalizations applied.
pub fn portable_frame_header(
    header: &Value,
    inputs: &[String],
    libm_dependent: &BTreeSet<String>,
) -> Value {
    let mut spellings: Vec<(String, String)> = inputs
        .iter()
        .filter_map(|spelling| {
            let name = file_name(spelling);
            if spelling.is_empty() || name.is_empty() || spelling == name {
                None
            } else {
                Some((spelling.clone(), name.to_owned()))
            }
        })
        .collect();
    // Longest first: a directory that is a prefix of another input's path
    // must not shadow the longer match.
    spellings.sort_by(|left, right| right.0.len().cmp(&left.0.len()));

    let mut masked = mask_input_paths(header, &spellings);
    if let Some(fields) = masked.get_mut("fields").and_then(Value::as_array_mut) {
        for descriptor in fields.iter_mut() {
            let Some(name) = descriptor
                .get("canonical_name")
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                continue;
            };
            if libm_dependent.contains(&name) {
                if let Some(object) = descriptor.as_object_mut() {
                    object.insert(
                        "data_reference".to_owned(),
                        Value::String(format!("libm:{name}")),
                    );
                }
            }
        }
    }
    if let Some(object) = masked.as_object_mut() {
        object.insert(
            "portable_rule".to_owned(),
            Value::String(PORTABLE_HEADER_RULE.to_owned()),
        );
    } else {
        let mut object = Map::new();
        object.insert("header".to_owned(), masked);
        masked = Value::Object(object);
    }
    masked
}

/// The sha256 of [`portable_frame_header`], canonically encoded.
pub fn portable_frame_header_sha256(
    header: &Value,
    inputs: &[String],
    libm_dependent: &BTreeSet<String>,
) -> String {
    crate::digest::bytes_sha256(
        crate::engine::canonical_json(&portable_frame_header(header, inputs, libm_dependent)).as_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn header(sample: &str, humidity: &str) -> Value {
        json!({
            "fields": [
                {
                    "canonical_name": "air_temperature",
                    "data_reference": "sha256:aaaa",
                    "source_field": format!("{sample}:t"),
                },
                {
                    "canonical_name": "specific_humidity",
                    "data_reference": format!("sha256:{humidity}"),
                    "source_field": format!("{sample}:r;{sample}:t"),
                },
            ],
        })
    }

    fn declared_libm() -> BTreeSet<String> {
        BTreeSet::from(["specific_humidity".to_owned()])
    }

    #[test]
    fn the_portable_digest_ignores_where_the_inputs_live() {
        let posix = header("/srv/inputs/staging/sample.nc", "bbbb");
        let windows = header("D:\\inputs\\staging\\sample.nc", "bbbb");
        assert_eq!(
            portable_frame_header_sha256(
                &posix,
                &["/srv/inputs/staging/sample.nc".to_owned()],
                &declared_libm(),
            ),
            portable_frame_header_sha256(
                &windows,
                &["D:\\inputs\\staging\\sample.nc".to_owned()],
                &declared_libm(),
            ),
        );
    }

    #[test]
    fn the_portable_digest_ignores_a_derived_field_s_last_bits() {
        let inputs = ["/staging/sample.nc".to_owned()];
        assert_eq!(
            portable_frame_header_sha256(
                &header("/staging/sample.nc", "bbbb"), &inputs, &declared_libm()),
            portable_frame_header_sha256(
                &header("/staging/sample.nc", "cccc"), &inputs, &declared_libm()),
        );
    }

    #[test]
    fn a_direct_field_s_bytes_still_move_the_portable_digest() {
        let inputs = ["/staging/sample.nc".to_owned()];
        let left = header("/staging/sample.nc", "bbbb");
        let mut right = left.clone();
        right["fields"][0]["data_reference"] = Value::String("sha256:zzzz".to_owned());
        assert_ne!(
            portable_frame_header_sha256(&left, &inputs, &declared_libm()),
            portable_frame_header_sha256(&right, &inputs, &declared_libm()),
        );
    }
}
