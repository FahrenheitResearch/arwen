//! Original checkpoint envelope for the complete resident NextGen state.
use crate::{
    Result,
    netcdf::{self, Attribute, Dataset, Dimension, Variable},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct Array {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl Array {
    pub fn size(&self) -> Result<usize> {
        let width = netcdf::datatype(&self.dtype)?.1;
        if width == 0 || self.dtype == "char" {
            return Err("checkpoint requires primitive numerical arrays".into());
        }
        self.shape.iter().try_fold(width, |n, d| {
            n.checked_mul(*d)
                .ok_or_else(|| "checkpoint array overflow".into())
        })
    }
}
fn schema(metadata: &serde_json::Value, arrays: &[Array], digest: &str) -> Result<Dataset> {
    let mut out = Dataset {
        format: "cdf5".into(),
        dimensions: vec![],
        variables: vec![],
        attributes: vec![
            Attribute::text("arwen_schema", "arwen.nextgen.checkpoint/v1"),
            Attribute::text("arwen_metadata", &serde_json::to_string(metadata)?),
            Attribute::text("arwen_arrays", &serde_json::to_string(arrays)?),
            Attribute::text("arwen_payload_sha256", digest),
        ],
    };
    let mut names = std::collections::HashSet::new();
    let mut offset = 0usize;
    for (i, a) in arrays.iter().enumerate() {
        if a.name.is_empty() || !names.insert(&a.name) {
            return Err("checkpoint array names must be unique".into());
        }
        let size = a.size()?;
        // An empty component roster is valid, but a zero NetCDF dimension
        // means unlimited. Preserve it in the inventory without a data variable.
        if size > 0 {
            let dims = a
                .shape
                .iter()
                .enumerate()
                .map(|(j, len)| {
                    let name = format!("d{i:06}_{j}");
                    out.dimensions.push(Dimension {
                        name: name.clone(),
                        len: *len,
                        unlimited: false,
                    });
                    name
                })
                .collect();
            out.variables.push(Variable {
                name: format!("a{i:06}"),
                dtype: a.dtype.clone(),
                dimensions: dims,
                attributes: vec![Attribute::text("resident_name", &a.name)],
                offset,
                strings: vec![],
            });
        }
        offset = offset
            .checked_add(size)
            .ok_or("checkpoint payload overflow")?;
    }
    Ok(out)
}
pub fn write(
    path: &Path,
    metadata: &serde_json::Value,
    arrays: &[Array],
    payload: &[u8],
) -> Result<serde_json::Value> {
    let digest = format!("{:x}", Sha256::digest(payload));
    let schema = schema(metadata, arrays, &digest)?;
    let mut receipt = netcdf::write(path, &schema, payload)?;
    receipt["operation"] = "checkpoint-write".into();
    receipt["sha256"] = digest.into();
    Ok(receipt)
}
pub fn read(path: &Path) -> Result<(serde_json::Value, Vec<u8>)> {
    let start = std::time::Instant::now();
    let (actual, payload) = netcdf::read(path)?;
    let get = |key: &str| -> Result<&str> {
        Ok(&actual
            .attributes
            .iter()
            .find(|a| a.name == key && a.dtype == "char")
            .ok_or("incomplete checkpoint envelope")?
            .text)
    };
    if get("arwen_schema")? != "arwen.nextgen.checkpoint/v1" {
        return Err("checkpoint schema differs".into());
    }
    let metadata: serde_json::Value = serde_json::from_str(get("arwen_metadata")?)?;
    let arrays: Vec<Array> = serde_json::from_str(get("arwen_arrays")?)?;
    let digest = format!("{:x}", Sha256::digest(&payload));
    if digest != get("arwen_payload_sha256")? {
        return Err("checkpoint payload SHA256 differs".into());
    }
    if actual != schema(&metadata, &arrays, &digest)? {
        return Err("checkpoint container differs from complete declared inventory".into());
    }
    Ok((
        serde_json::json!({"operation":"checkpoint-read","metadata":metadata,"arrays":arrays,"sha256":digest,"read_validate_ms":start.elapsed().as_secs_f64()*1000.}),
        payload,
    ))
}
