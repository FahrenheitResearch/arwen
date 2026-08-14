//! Typed access to the ODIM attribute groups (`what`, `where`, `how`).
//!
//! ODIM states its metadata almost entirely as HDF5 attributes, and different
//! national writers spell the same attribute with different HDF5 datatypes:
//! `nbins` arrives as `i64` from one vendor and `u32` from another, `elangle`
//! as `f32` or `f64`, and scalars are sometimes written as one-element arrays.
//! Reading those with a fixed type would make the decoder vendor-specific, so
//! everything numeric is normalised through `f64` here, exactly as
//! `rustwx-io`'s `hdf5_dataset_values_f64` does for payloads.
//!
//! [`Attrs`] reads a group's attributes once and answers by name, so a group
//! that is consulted for a dozen fields is parsed once and every "missing
//! attribute" error can name the group it was missing from.

use hdf5_reader::Attribute;
use hdf5_reader::group::Group;
use hdf5_reader::messages::datatype::Datatype;

use crate::error::{OdimError, Result};

/// The attributes of one ODIM group, read once and queried by name.
#[derive(Debug)]
pub struct Attrs {
    path: String,
    items: Vec<Attribute>,
}

impl Attrs {
    /// Read every attribute of `group`. `path` is the ODIM path used in error
    /// messages, e.g. `/dataset3/data1/what`.
    pub fn read(path: impl Into<String>, group: &Group) -> Result<Self> {
        let path = path.into();
        let items = group.attributes().map_err(|err| OdimError::Format {
            context: format!("reading attributes of {path}"),
            detail: err.to_string(),
        })?;
        Ok(Attrs { path, items })
    }

    /// An empty set, for an optional group that is absent. Every lookup then
    /// reports "missing", which is the truth.
    pub fn empty(path: impl Into<String>) -> Self {
        Attrs {
            path: path.into(),
            items: Vec::new(),
        }
    }

    /// The ODIM path these attributes came from.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Attribute names present, sorted. Used by `rw_odim inspect` so an
    /// unfamiliar national writer can be surveyed without a code change.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.items.iter().map(|a| a.name.clone()).collect();
        names.sort();
        names
    }

    fn find(&self, name: &str) -> Option<&Attribute> {
        self.items.iter().find(|a| a.name == name)
    }

    fn missing(&self, name: &str) -> OdimError {
        OdimError::MissingAttribute {
            group: self.path.clone(),
            name: name.to_string(),
        }
    }

    /// Whether the attribute is present at all.
    pub fn has(&self, name: &str) -> bool {
        self.find(name).is_some()
    }

    /// Every element of a numeric attribute, widened to `f64`.
    ///
    /// Widening is lossless for every ODIM-legal storage type except `u64`/`i64`
    /// beyond 2^53, which no ODIM attribute uses.
    pub fn f64_vec_opt(&self, name: &str) -> Result<Option<Vec<f64>>> {
        let Some(attr) = self.find(name) else {
            return Ok(None);
        };
        self.values_f64(attr).map(Some)
    }

    /// Every element of a required numeric attribute.
    pub fn f64_vec(&self, name: &str) -> Result<Vec<f64>> {
        self.f64_vec_opt(name)?.ok_or_else(|| self.missing(name))
    }

    fn values_f64(&self, attr: &Attribute) -> Result<Vec<f64>> {
        let mismatch = |expected: &str| OdimError::AttributeType {
            group: self.path.clone(),
            name: attr.name.clone(),
            found: format!("{:?}", attr.datatype),
            expected: expected.to_string(),
        };
        let widen = |values: Result<Vec<f64>>| values;
        match &attr.datatype {
            Datatype::FloatingPoint { size, .. } => match size {
                4 => widen(
                    attr.read_1d::<f32>()
                        .map(|v| v.into_iter().map(f64::from).collect())
                        .map_err(|_| mismatch("f32")),
                ),
                8 => attr.read_1d::<f64>().map_err(|_| mismatch("f64")),
                _ => Err(mismatch("a 4- or 8-byte float")),
            },
            Datatype::FixedPoint { size, signed, .. } => match (size, signed) {
                (1, true) => attr
                    .read_1d::<i8>()
                    .map(|v| v.into_iter().map(f64::from).collect())
                    .map_err(|_| mismatch("i8")),
                (1, false) => attr
                    .read_1d::<u8>()
                    .map(|v| v.into_iter().map(f64::from).collect())
                    .map_err(|_| mismatch("u8")),
                (2, true) => attr
                    .read_1d::<i16>()
                    .map(|v| v.into_iter().map(f64::from).collect())
                    .map_err(|_| mismatch("i16")),
                (2, false) => attr
                    .read_1d::<u16>()
                    .map(|v| v.into_iter().map(f64::from).collect())
                    .map_err(|_| mismatch("u16")),
                (4, true) => attr
                    .read_1d::<i32>()
                    .map(|v| v.into_iter().map(f64::from).collect())
                    .map_err(|_| mismatch("i32")),
                (4, false) => attr
                    .read_1d::<u32>()
                    .map(|v| v.into_iter().map(f64::from).collect())
                    .map_err(|_| mismatch("u32")),
                (8, true) => attr
                    .read_1d::<i64>()
                    .map(|v| v.into_iter().map(|x| x as f64).collect())
                    .map_err(|_| mismatch("i64")),
                (8, false) => attr
                    .read_1d::<u64>()
                    .map(|v| v.into_iter().map(|x| x as f64).collect())
                    .map_err(|_| mismatch("u64")),
                _ => Err(mismatch("a 1-, 2-, 4- or 8-byte integer")),
            },
            _ => Err(mismatch("a numeric type")),
        }
    }

    /// A required numeric scalar.
    ///
    /// A one-element array satisfies this: several national writers spell
    /// scalars that way and ODIM does not forbid it.
    pub fn f64(&self, name: &str) -> Result<f64> {
        self.f64_opt(name)?.ok_or_else(|| self.missing(name))
    }

    /// An optional numeric scalar.
    pub fn f64_opt(&self, name: &str) -> Result<Option<f64>> {
        let Some(values) = self.f64_vec_opt(name)? else {
            return Ok(None);
        };
        match values.len() {
            1 => Ok(Some(values[0])),
            n => Err(OdimError::AttributeType {
                group: self.path.clone(),
                name: name.to_string(),
                found: format!("{n} elements"),
                expected: "a scalar".to_string(),
            }),
        }
    }

    /// A required count, refused if it is negative or not a whole number.
    pub fn usize(&self, name: &str) -> Result<usize> {
        let value = self.f64(name)?;
        if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
            return Err(OdimError::AttributeType {
                group: self.path.clone(),
                name: name.to_string(),
                found: format!("{value}"),
                expected: "a non-negative whole number".to_string(),
            });
        }
        Ok(value as usize)
    }

    /// An optional count.
    pub fn usize_opt(&self, name: &str) -> Result<Option<usize>> {
        if self.has(name) {
            self.usize(name).map(Some)
        } else {
            Ok(None)
        }
    }

    /// A required string attribute, trailing NULs and spaces trimmed.
    pub fn string(&self, name: &str) -> Result<String> {
        self.string_opt(name)?.ok_or_else(|| self.missing(name))
    }

    /// An optional string attribute.
    pub fn string_opt(&self, name: &str) -> Result<Option<String>> {
        let Some(attr) = self.find(name) else {
            return Ok(None);
        };
        let raw = attr.read_string().map_err(|err| OdimError::AttributeType {
            group: self.path.clone(),
            name: name.to_string(),
            found: format!("{:?} ({err})", attr.datatype),
            expected: "a string".to_string(),
        })?;
        Ok(Some(raw.trim_end_matches(['\0', ' ']).to_string()))
    }
}

/// Fetch a child group, returning `None` when it is simply absent.
///
/// `how` is optional everywhere in ODIM, so "absent" must not be an error;
/// but a `how` that exists and will not parse must be, which is why this
/// cannot just swallow every failure.
pub fn optional_group(parent: &Group, name: &str) -> Option<Group> {
    parent.group(name).ok()
}

/// The trailing integer of a group name like `dataset12` or `data3`.
///
/// Returns `None` for a name that does not fit the pattern, which is how a
/// vendor extension group sitting beside the datasets gets skipped rather
/// than mistaken for a sweep.
pub fn indexed_suffix(name: &str, prefix: &str) -> Option<usize> {
    let rest = name.strip_prefix(prefix)?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_groups_are_recognised_and_ordered_numerically() {
        assert_eq!(indexed_suffix("dataset1", "dataset"), Some(1));
        assert_eq!(indexed_suffix("dataset12", "dataset"), Some(12));
        assert_eq!(indexed_suffix("data3", "data"), Some(3));
    }

    #[test]
    fn non_sweep_siblings_are_not_mistaken_for_sweeps() {
        // `dataset` with no index, and the vendor extension groups that sit
        // beside the datasets, must not be read as sweeps.
        assert_eq!(indexed_suffix("dataset", "dataset"), None);
        assert_eq!(indexed_suffix("datasetX", "dataset"), None);
        assert_eq!(indexed_suffix("how", "dataset"), None);
        assert_eq!(indexed_suffix("what", "dataset"), None);
        assert_eq!(indexed_suffix("where", "dataset"), None);
        // `data1/what` is a group under a data group, not a data group.
        assert_eq!(indexed_suffix("what", "data"), None);
        // ...but `dataset1` must not be picked up when scanning for `data`.
        assert_eq!(indexed_suffix("dataset1", "data"), None);
    }

    #[test]
    fn the_numeric_suffix_is_what_orders_sweeps_not_the_string() {
        // The bug this guards: "dataset10" < "dataset2" lexically, so a
        // lexical sort would reorder a 12-sweep Romanian volume.
        let mut names = vec!["dataset10", "dataset2", "dataset1", "dataset12"];
        names.sort();
        assert_eq!(
            names,
            vec!["dataset1", "dataset10", "dataset12", "dataset2"]
        );
        let mut indexed: Vec<usize> = names
            .iter()
            .filter_map(|n| indexed_suffix(n, "dataset"))
            .collect();
        indexed.sort_unstable();
        assert_eq!(indexed, vec![1, 2, 10, 12]);
    }

    #[test]
    fn an_absent_group_reports_every_attribute_as_missing() {
        let attrs = Attrs::empty("/dataset1/how");
        assert!(!attrs.has("NI"));
        assert!(attrs.names().is_empty());
        let err = attrs.f64("NI").unwrap_err();
        assert!(
            matches!(&err, OdimError::MissingAttribute { group, name }
                if group == "/dataset1/how" && name == "NI"),
            "{err}"
        );
        // Optional lookups stay quiet, which is what makes `how` optional.
        assert_eq!(attrs.f64_opt("NI").unwrap(), None);
        assert_eq!(attrs.string_opt("task").unwrap(), None);
    }
}
