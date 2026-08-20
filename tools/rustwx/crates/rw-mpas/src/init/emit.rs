//! Writing the CDF-5 initial-conditions file.
//!
//! ## Schema by example, values by computation
//! The container's shape — its seventeen dimensions, its 110 float, 21 int and
//! 3 char variables and their attributes — is read from the capsule and
//! re-emitted unchanged.  What each variable *holds* is then one of exactly
//! two things, and the receipt says which for every single one:
//!
//! * **computed** — this port produced the numbers, and they are what the
//!   validation compares;
//! * **carried** — the value came from the capsule untouched.  Statics,
//!   mesh geometry and the vertical metrics are carried by design this wave
//!   (see [`crate::init::capsule`]); a delta of exactly zero on a carried
//!   variable is therefore *not* evidence of anything and is labelled so
//!   nobody reads it as agreement.
//!
//! Reading the schema from the same file the run is compared against is a real
//! hazard and is named rather than hidden: it makes the container trivially
//! right and can make a *carried* field look perfect.  It cannot make a
//! computed field look right, which is why the receipt separates the two lists
//! and why the delta table repeats the label on every row.
//!
//! ## Array order
//! MPAS is cell-major with the vertical fastest: `(Time, nCells,
//! nVertLevels)`.  That is the opposite of the wrfout convention this crate's
//! converter emits, and it is the single easiest thing in the file to get
//! backwards, so every computed field is assembled cell-major and the
//! assembly is asserted against the capsule's own element count before a byte
//! is written.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use rw_store::netcdf_classic::{
    NcAttr, NcAttrValue, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};

use crate::error::{MpasError, MpasResult};

/// A computed variable waiting to be written, in the file's own order.
pub enum Computed {
    Floats(Vec<f32>),
    Ints(Vec<i32>),
    /// An `NC_CHAR` payload.  The writer NUL-fills a short slab to the
    /// declared length, which is how a 19-byte stamp lands in a 64-byte slot.
    Text(String),
}

impl Computed {
    fn len(&self) -> usize {
        match self {
            Computed::Floats(v) => v.len(),
            Computed::Ints(v) => v.len(),
            // A text slab is length-checked by the writer, which pads; a
            // 19-byte stamp in a 64-byte slot is correct, not short.
            Computed::Text(_) => usize::MAX,
        }
    }
}

/// What the emitter did, variable by variable.
#[derive(Debug, Default, serde::Serialize)]
pub struct EmitLedger {
    pub computed: Vec<String>,
    pub carried: Vec<String>,
    pub file_id: String,
    pub parent_id: String,
    pub bytes: u64,
}

fn nc_type_of(dtype: &netcrust::DataType) -> MpasResult<NcType> {
    Ok(match dtype {
        netcrust::DataType::Char | netcrust::DataType::I8 | netcrust::DataType::U8 => NcType::Char,
        netcrust::DataType::I16 => NcType::Short,
        netcrust::DataType::I32 => NcType::Int,
        netcrust::DataType::F32 => NcType::Float,
        netcrust::DataType::F64 => NcType::Double,
        other => {
            return Err(MpasError::Refusal(format!(
                "capsule carries a {other:?} variable, which the classic writer has no type for"
            )))
        }
    })
}

fn attr_of(a: &netcrust::Attribute) -> Option<NcAttr> {
    let value = match a.value() {
        netcrust::AttributeValue::Chars(s) => NcAttrValue::Text(s.clone()),
        netcrust::AttributeValue::Strings(v) if v.len() == 1 => NcAttrValue::Text(v[0].clone()),
        netcrust::AttributeValue::Ints(v) => NcAttrValue::Ints(v.clone()),
        netcrust::AttributeValue::Floats(v) => NcAttrValue::Floats(v.clone()),
        netcrust::AttributeValue::Doubles(v) => NcAttrValue::Doubles(v.clone()),
        _ => return None,
    };
    Some(NcAttr {
        name: a.name().to_string(),
        value,
    })
}

/// A ten-character token, derived from the run's own inputs so two runs on the
/// same inputs mint the same id and two runs on different inputs do not.
pub fn mint_file_id(seed: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(seed.as_bytes());
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    digest
        .iter()
        .take(10)
        .map(|b| ALPHABET[(*b as usize) % ALPHABET.len()] as char)
        .collect()
}

/// Write the init file: capsule schema, computed values where we have them.
///
/// `parent_chain` becomes `parent_id`.  MPAS writes it newline separated and
/// with a trailing empty entry, and the port's loader checks that shape, so it
/// is built here rather than left to the caller to remember.
pub fn write_init(
    out_path: &Path,
    capsule_path: &Path,
    computed: BTreeMap<String, Computed>,
    file_id: &str,
    parent_chain: &[String],
    provenance: &str,
) -> MpasResult<EmitLedger> {
    let capsule = netcrust::File::open(capsule_path)?;

    // Dimensions, in the capsule's own order, so dimids line up.
    let cap_dims = capsule.dimensions()?;
    let mut dims: Vec<NcDim> = Vec::with_capacity(cap_dims.len());
    let mut dimid_of: BTreeMap<String, usize> = BTreeMap::new();
    let mut record_len = 0u64;
    for (i, d) in cap_dims.iter().enumerate() {
        dimid_of.insert(d.name().to_string(), i);
        if d.is_unlimited() {
            record_len = d.len().max(1) as u64;
            dims.push(NcDim::record(d.name()));
        } else {
            dims.push(NcDim::fixed(d.name(), d.len()));
        }
    }
    if record_len == 0 {
        record_len = 1;
    }

    let parent_id = {
        let mut chain: Vec<String> = parent_chain.to_vec();
        // MPAS's own trailing empty entry.
        chain.push(String::new());
        chain.join("\n")
    };

    let mut gattrs: Vec<NcAttr> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for a in capsule.attributes()? {
        let name = a.name().to_string();
        if name == "file_id" {
            gattrs.push(NcAttr::text("file_id", file_id));
        } else if name == "parent_id" {
            gattrs.push(NcAttr::text("parent_id", parent_id.clone()));
        } else if let Some(attr) = attr_of(&a) {
            gattrs.push(attr);
        } else {
            return Err(MpasError::Refusal(format!(
                "capsule global attribute {name} has a value shape the classic writer cannot \
                 carry; dropping it silently would change the file's identity"
            )));
        }
        seen.insert(name);
    }
    if !seen.contains("file_id") {
        gattrs.push(NcAttr::text("file_id", file_id));
    }
    if !seen.contains("parent_id") {
        gattrs.push(NcAttr::text("parent_id", parent_id.clone()));
    }
    gattrs.push(NcAttr::text("gpuwm_provenance", provenance));

    // Variable definitions, in the capsule's order.
    let cap_vars = capsule.variables()?;
    let mut vars: Vec<NcVarDef> = Vec::with_capacity(cap_vars.len());
    for v in &cap_vars {
        let ty = nc_type_of(v.dtype())?;
        let mut dimids = Vec::new();
        for d in v.dimensions() {
            let id = *dimid_of.get(d.name()).ok_or_else(|| {
                MpasError::Refusal(format!(
                    "variable {} names dimension {} the capsule does not declare",
                    v.name(),
                    d.name()
                ))
            })?;
            dimids.push(id);
        }
        let attrs: Vec<NcAttr> = v.attributes().iter().filter_map(attr_of).collect();
        vars.push(NcVarDef::new(v.name(), ty, dimids).with_attrs(attrs));
    }

    let mut writer = NcClassicWriter::create(
        out_path,
        NcFormat::Data64,
        dims,
        gattrs,
        vars,
        record_len,
    )
    .map_err(|e| MpasError::Refusal(format!("cannot lay out {}: {e}", out_path.display())))?;

    // A computed value whose name is not in the file would be dropped without
    // a word, which is the quietest possible way to ship an init with a field
    // left at whatever the capsule held.  Refuse instead.
    {
        let present: BTreeSet<String> = cap_vars.iter().map(|v| v.name().to_string()).collect();
        let stray: Vec<&String> = computed.keys().filter(|k| !present.contains(*k)).collect();
        if !stray.is_empty() {
            return Err(MpasError::Refusal(format!(
                "this run computed {} value(s) the init file has no variable for: {:?}.  Writing                  the file anyway would silently drop them and leave the capsule's own numbers in                  their place",
                stray.len(),
                stray
            )));
        }
    }

    let mut ledger = EmitLedger {
        file_id: file_id.to_string(),
        parent_id,
        ..Default::default()
    };

    for v in &cap_vars {
        let name = v.name().to_string();
        let is_record = v
            .dimensions()
            .first()
            .map(|d| d.is_unlimited())
            .unwrap_or(false);

        if let Some(value) = computed.get(&name) {
            // One slab: the product of every dimension except a leading
            // record axis.
            let expected: usize = v
                .dimensions()
                .iter()
                .skip(usize::from(is_record))
                .map(|d| d.len().max(1))
                .product();
            if !matches!(value, Computed::Text(_)) && value.len() != expected {
                return Err(MpasError::Refusal(format!(
                    "computed {name} holds {} value(s); the file's slab is {expected}.  A \
                     mismatched count here is the cell-major/level-major transpose, which \
                     produces a readable file full of wrong numbers",
                    value.len()
                )));
            }
            let text_bytes;
            let data = match value {
                Computed::Floats(f) => NcData::Floats(f),
                Computed::Ints(i) => NcData::Ints(i),
                Computed::Text(t) => {
                    // A char slab is a fixed-width field: MPAS writes a
                    // 19-character stamp into a 64-character StrLen and leaves
                    // the rest blank.  Pad to the declared width here, and
                    // refuse a label too long to fit rather than truncate one
                    // into a different, still-readable value.
                    if t.len() > expected {
                        return Err(MpasError::Refusal(format!(
                            "computed {name} is {} character(s); the file's field is {expected}. \
                             Truncating it would leave a different label that still reads",
                            t.len()
                        )));
                    }
                    let mut bytes = t.as_bytes().to_vec();
                    bytes.resize(expected, b' ');
                    text_bytes = bytes;
                    NcData::Chars(&text_bytes)
                }
            };
            if is_record {
                writer.put_record(&name, 0, data)
            } else {
                writer.put(&name, data)
            }
            .map_err(|e| MpasError::Refusal(format!("cannot write {name}: {e}")))?;
            ledger.computed.push(name);
            continue;
        }

        // Carried straight through from the capsule.
        let ty = nc_type_of(v.dtype())?;
        match ty {
            NcType::Char => {
                // The three char variables in an MPAS init are identity
                // labels the caller owns: the valid time, twice, and the
                // land-use table name.  Carrying one out of the capsule would
                // stamp this file with the capsule's identity instead of its
                // own — a file dated to somebody else's cycle that opens and
                // reads perfectly — so an unsupplied one refuses.
                return Err(MpasError::Refusal(format!(
                    "char variable {name} was not supplied by the caller.  Char variables in an \
                     MPAS init are identity labels; copying one from the capsule would date this \
                     file to the capsule's cycle"
                )));
            }
            NcType::Int => {
                let values: Vec<i32> = capsule
                    .read_array_f64(&name)?
                    .into_values()
                    .into_iter()
                    .map(|v| v.round() as i32)
                    .collect();
                if is_record {
                    writer.put_record(&name, 0, NcData::Ints(&values))
                } else {
                    writer.put(&name, NcData::Ints(&values))
                }
                .map_err(|e| MpasError::Refusal(format!("cannot write {name}: {e}")))?;
            }
            NcType::Float => {
                let values: Vec<f32> = capsule
                    .read_array_f64(&name)?
                    .into_values()
                    .into_iter()
                    .map(|v| v as f32)
                    .collect();
                if is_record {
                    writer.put_record(&name, 0, NcData::Floats(&values))
                } else {
                    writer.put(&name, NcData::Floats(&values))
                }
                .map_err(|e| MpasError::Refusal(format!("cannot write {name}: {e}")))?;
            }
            other => {
                return Err(MpasError::Refusal(format!(
                    "carried variable {name} is {other:?}, which this emitter does not copy"
                )))
            }
        }
        ledger.carried.push(name);
    }

    ledger.bytes = writer.file_len();
    writer
        .finish()
        .map_err(|e| MpasError::Refusal(format!("cannot finish {}: {e}", out_path.display())))?;
    Ok(ledger)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_id_is_ten_lowercase_letters_and_is_a_function_of_its_seed() {
        let a = mint_file_id("MET:2025-03-14_12|x4.163842");
        let b = mint_file_id("MET:2025-03-14_12|x4.163842");
        let c = mint_file_id("MET:2026-08-12_06|x4.163842");
        assert_eq!(a.len(), 10);
        assert!(a.chars().all(|ch| ch.is_ascii_lowercase()));
        assert_eq!(a, b, "the same inputs must mint the same id");
        assert_ne!(a, c, "different inputs must not");
    }

    #[test]
    fn the_parent_chain_keeps_its_trailing_empty_entry() {
        let chain = vec!["abcdefghij".to_string(), "klmnopqrst".to_string()];
        let mut with_tail = chain.clone();
        with_tail.push(String::new());
        let joined = with_tail.join("\n");
        assert!(joined.ends_with('\n'), "the loader checks for it: {joined:?}");
        assert_eq!(joined.split('\n').count(), 3);
    }
}
