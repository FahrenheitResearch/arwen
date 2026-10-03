//! Classic netCDF writer: the 64-bit offset format (CDF-2), as the netCDF
//! Users Guide ("File Format Specification", classic and 64-bit offset
//! formats) defines it.
//!
//! Fixed-size dimensions only (no record dimension): the header, then
//! every variable's values in definition order, big-endian, each padded to
//! four bytes. Output is deterministic.

use super::CfWriteError;

/// Values of an attribute or variable in one classic netCDF type.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Nc3Values {
    /// `byte`: 8-bit signed integers.
    Byte(Vec<i8>),
    /// `char`: 8-bit characters.
    Char(Vec<u8>),
    /// `short`: 16-bit signed integers.
    Short(Vec<i16>),
    /// `int`: 32-bit signed integers.
    Int(Vec<i32>),
    /// `float`: IEEE binary32.
    Float(Vec<f32>),
    /// `double`: IEEE binary64.
    Double(Vec<f64>),
}

impl Nc3Values {
    /// Text as `char` values.
    pub fn text(text: &str) -> Self {
        Self::Char(text.as_bytes().to_vec())
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::Byte(v) => v.len(),
            Self::Char(v) => v.len(),
            Self::Short(v) => v.len(),
            Self::Int(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::Double(v) => v.len(),
        }
    }

    /// `true` when there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn type_code(&self) -> u32 {
        match self {
            Self::Byte(_) => 1,
            Self::Char(_) => 2,
            Self::Short(_) => 3,
            Self::Int(_) => 4,
            Self::Float(_) => 5,
            Self::Double(_) => 6,
        }
    }

    fn element_size(&self) -> usize {
        match self {
            Self::Byte(_) | Self::Char(_) => 1,
            Self::Short(_) => 2,
            Self::Int(_) | Self::Float(_) => 4,
            Self::Double(_) => 8,
        }
    }

    /// Big-endian bytes, padded with zeros to a multiple of four.
    fn encode(&self, out: &mut Vec<u8>) {
        let start = out.len();
        match self {
            Self::Byte(v) => out.extend(v.iter().map(|x| *x as u8)),
            Self::Char(v) => out.extend_from_slice(v),
            Self::Short(v) => v
                .iter()
                .for_each(|x| out.extend_from_slice(&x.to_be_bytes())),
            Self::Int(v) => v
                .iter()
                .for_each(|x| out.extend_from_slice(&x.to_be_bytes())),
            Self::Float(v) => v
                .iter()
                .for_each(|x| out.extend_from_slice(&x.to_be_bytes())),
            Self::Double(v) => v
                .iter()
                .for_each(|x| out.extend_from_slice(&x.to_be_bytes())),
        }
        let written = out.len() - start;
        out.resize(start + padded(written), 0);
    }

    fn byte_len(&self) -> usize {
        self.len() * self.element_size()
    }
}

fn padded(len: usize) -> usize {
    len.div_ceil(4) * 4
}

/// A variable to write.
#[derive(Clone, Debug, PartialEq)]
pub struct Nc3Variable {
    /// Variable name.
    pub name: String,
    /// Dimension ids from [`Nc3Writer::add_dim`], outermost first.
    pub dims: Vec<usize>,
    /// Attributes in order.
    pub attrs: Vec<(String, Nc3Values)>,
    /// Values, row-major; as many as the dimensions hold.
    pub values: Nc3Values,
}

/// Builds a classic netCDF (CDF-2) file.
#[derive(Default)]
pub struct Nc3Writer {
    dims: Vec<(String, u64)>,
    attrs: Vec<(String, Nc3Values)>,
    vars: Vec<Nc3Variable>,
}

/// netCDF-C's name rule (`NC_check_name`): not empty, no `/`, no control
/// characters, no trailing space, and a first character that is a letter,
/// a digit, `_` or non-ASCII.
pub fn is_valid_name(name: &str) -> bool {
    recast_radar_hdf5::write::netcdf4::is_valid_name(name)
}

/// `name` made valid for netCDF: invalid characters become `_`, and a name
/// that does not start with a letter, digit, `_` or non-ASCII character is
/// prefixed with `_`.
pub fn sanitize_name(name: &str) -> String {
    if is_valid_name(name) {
        return name.to_owned();
    }
    let mut out: String = name
        .chars()
        .map(|c| if c == '/' || c.is_control() { '_' } else { c })
        .collect();
    while out.ends_with(' ') {
        out.pop();
        out.push('_');
    }
    match out.chars().next() {
        Some(first) if first.is_ascii_alphanumeric() || first == '_' || !first.is_ascii() => {}
        _ => out.insert(0, '_'),
    }
    out
}

impl Nc3Writer {
    /// An empty file.
    pub fn new() -> Self {
        Self::default()
    }

    /// Define a dimension (length at least 1) and return its id.
    pub fn add_dim(&mut self, name: &str, len: u64) -> Result<usize, CfWriteError> {
        if !is_valid_name(name) {
            return Err(CfWriteError::Invalid(format!("dimension name {name:?}")));
        }
        if len == 0 || len > u64::from(u32::MAX) {
            return Err(CfWriteError::Invalid(format!(
                "dimension {name} of length {len} (1 to 4,294,967,295 in a classic file)"
            )));
        }
        if self.dims.iter().any(|(existing, _)| existing == name) {
            return Err(CfWriteError::Invalid(format!("duplicate dimension {name}")));
        }
        self.dims.push((name.to_owned(), len));
        Ok(self.dims.len() - 1)
    }

    /// The id of dimension `name`.
    pub fn dim(&self, name: &str) -> Option<usize> {
        self.dims.iter().position(|(existing, _)| existing == name)
    }

    /// Length of dimension `id`.
    pub fn dim_len(&self, id: usize) -> Option<u64> {
        self.dims.get(id).map(|(_, len)| *len)
    }

    /// Add a global attribute.
    pub fn add_attr(&mut self, name: &str, values: Nc3Values) -> Result<(), CfWriteError> {
        check_attr(name, &self.attrs)?;
        self.attrs.push((name.to_owned(), values));
        Ok(())
    }

    /// `true` when a global attribute `name` exists.
    pub fn has_attr(&self, name: &str) -> bool {
        self.attrs.iter().any(|(existing, _)| existing == name)
    }

    /// `true` when a variable `name` exists.
    pub fn has_var(&self, name: &str) -> bool {
        self.vars.iter().any(|var| var.name == name)
    }

    /// Add a variable.
    pub fn add_var(&mut self, var: Nc3Variable) -> Result<(), CfWriteError> {
        if !is_valid_name(&var.name) {
            return Err(CfWriteError::Invalid(format!(
                "variable name {:?}",
                var.name
            )));
        }
        if self.has_var(&var.name) {
            return Err(CfWriteError::Invalid(format!(
                "duplicate variable {}",
                var.name
            )));
        }
        let mut expected = 1u64;
        for dim in &var.dims {
            let len = self.dim_len(*dim).ok_or_else(|| {
                CfWriteError::Invalid(format!("variable {}: dimension {dim} undefined", var.name))
            })?;
            expected = expected.saturating_mul(len);
        }
        if var.values.len() as u64 != expected {
            return Err(CfWriteError::Invalid(format!(
                "variable {}: {} values for {expected} elements",
                var.name,
                var.values.len()
            )));
        }
        let mut seen: Vec<(String, Nc3Values)> = Vec::with_capacity(var.attrs.len());
        for (name, values) in &var.attrs {
            check_attr(name, &seen)?;
            seen.push((name.clone(), values.clone()));
        }
        self.vars.push(var);
        Ok(())
    }

    /// Lay the file out and return its bytes.
    pub fn finish(self) -> Result<Vec<u8>, CfWriteError> {
        let mut header = Vec::new();
        self.encode_header(&mut header, &vec![0; self.vars.len()]);
        let mut offsets = Vec::with_capacity(self.vars.len());
        let mut next = header.len() as u64;
        for var in &self.vars {
            offsets.push(next);
            next += padded(var.values.byte_len()) as u64;
        }
        let total = usize::try_from(next)
            .map_err(|_| CfWriteError::TooLarge("file larger than addressable memory".into()))?;
        let mut out = Vec::with_capacity(total);
        self.encode_header(&mut out, &offsets);
        for var in &self.vars {
            var.values.encode(&mut out);
        }
        if out.len() != total {
            return Err(CfWriteError::Invalid(format!(
                "laid out {total} bytes but wrote {}",
                out.len()
            )));
        }
        Ok(out)
    }

    fn encode_header(&self, out: &mut Vec<u8>, offsets: &[u64]) {
        out.extend_from_slice(b"CDF\x02");
        out.extend_from_slice(&0u32.to_be_bytes());
        // Dimensions.
        if self.dims.is_empty() {
            out.extend_from_slice(&[0; 8]);
        } else {
            out.extend_from_slice(&0x0Au32.to_be_bytes());
            out.extend_from_slice(&(self.dims.len() as u32).to_be_bytes());
            for (name, len) in &self.dims {
                encode_name(out, name);
                out.extend_from_slice(&(*len as u32).to_be_bytes());
            }
        }
        encode_attrs(out, &self.attrs);
        if self.vars.is_empty() {
            out.extend_from_slice(&[0; 8]);
        } else {
            out.extend_from_slice(&0x0Bu32.to_be_bytes());
            out.extend_from_slice(&(self.vars.len() as u32).to_be_bytes());
            for (var, offset) in self.vars.iter().zip(offsets) {
                encode_name(out, &var.name);
                out.extend_from_slice(&(var.dims.len() as u32).to_be_bytes());
                for dim in &var.dims {
                    out.extend_from_slice(&(*dim as u32).to_be_bytes());
                }
                encode_attrs(out, &var.attrs);
                out.extend_from_slice(&var.values.type_code().to_be_bytes());
                // vsize saturates for a variable of 4 GiB or more.
                let vsize = u32::try_from(padded(var.values.byte_len())).unwrap_or(u32::MAX);
                out.extend_from_slice(&vsize.to_be_bytes());
                out.extend_from_slice(&offset.to_be_bytes());
            }
        }
    }
}

fn check_attr(name: &str, existing: &[(String, Nc3Values)]) -> Result<(), CfWriteError> {
    if !is_valid_name(name) {
        return Err(CfWriteError::Invalid(format!("attribute name {name:?}")));
    }
    if existing.iter().any(|(have, _)| have == name) {
        return Err(CfWriteError::Invalid(format!("duplicate attribute {name}")));
    }
    Ok(())
}

fn encode_name(out: &mut Vec<u8>, name: &str) {
    out.extend_from_slice(&(name.len() as u32).to_be_bytes());
    out.extend_from_slice(name.as_bytes());
    out.resize(out.len() + padded(name.len()) - name.len(), 0);
}

fn encode_attrs(out: &mut Vec<u8>, attrs: &[(String, Nc3Values)]) {
    if attrs.is_empty() {
        out.extend_from_slice(&[0; 8]);
        return;
    }
    out.extend_from_slice(&0x0Cu32.to_be_bytes());
    out.extend_from_slice(&(attrs.len() as u32).to_be_bytes());
    for (name, values) in attrs {
        encode_name(out, name);
        out.extend_from_slice(&values.type_code().to_be_bytes());
        out.extend_from_slice(&(values.len() as u32).to_be_bytes());
        values.encode(out);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::netcdf3::{IntKind, Nc3File, NcArray, NcValue};

    fn attr(value: &NcValue) -> Nc3Values {
        match value {
            NcValue::Str(text) => Nc3Values::text(text),
            NcValue::Floats(values) => Nc3Values::Float(values.clone()),
            NcValue::Doubles(values) => Nc3Values::Double(values.clone()),
            NcValue::Ints(values, IntKind::I8) => {
                Nc3Values::Byte(values.iter().map(|v| *v as i8).collect())
            }
            NcValue::Ints(values, IntKind::I16) => {
                Nc3Values::Short(values.iter().map(|v| *v as i16).collect())
            }
            NcValue::Ints(values, _) => Nc3Values::Int(values.iter().map(|v| *v as i32).collect()),
            NcValue::Strings(_) => panic!("netCDF-4 strings in a classic file"),
        }
    }

    fn array(values: NcArray) -> Nc3Values {
        match values {
            NcArray::I8(v) => Nc3Values::Byte(v),
            NcArray::Char(v) => Nc3Values::Char(v),
            NcArray::I16(v) => Nc3Values::Short(v),
            NcArray::I32(v) => Nc3Values::Int(v),
            NcArray::F32(v) => Nc3Values::Float(v),
            NcArray::F64(v) => Nc3Values::Double(v),
            other => panic!("not a classic type: {other:?}"),
        }
    }

    /// The stored bytes (NaN payloads included).
    fn bytes(values: &Nc3Values) -> (u32, Vec<u8>) {
        let mut out = Vec::new();
        values.encode(&mut out);
        (values.type_code(), out)
    }

    /// Real classic CfRadial files (CDF-1, X-SAPR and DOW8) re-encoded
    /// through `Nc3Writer`: every dimension, attribute (in file order) and
    /// variable reads back with the same type and value bytes. The files
    /// themselves differ: the CDF-1 sources are written as CDF-2 (64-bit
    /// offsets), so this is not a byte-for-byte copy of the file.
    #[test]
    fn real_classic_files_reencode_with_the_same_types_and_values() {
        for id in [
            "cfrad1-xsapr-sgp-20110520-ppi-classic",
            "cfrad1-dow8-20211011-223602-rhi-trim3-classic",
        ] {
            let source = recast_radar_testdata::bytes(id).unwrap();
            let file = Nc3File::open(&source).unwrap();
            let mut nc = Nc3Writer::new();
            for (name, len) in &file.dims {
                nc.add_dim(name, *len as u64).unwrap();
            }
            for (name, value) in &file.gattrs {
                nc.add_attr(name, attr(value)).unwrap();
            }
            let mut vars: Vec<_> = file.vars.values().collect();
            vars.sort_by_key(|var| var.index);
            for var in &vars {
                nc.add_var(Nc3Variable {
                    name: var.name.clone(),
                    dims: var.dim_ids.clone(),
                    attrs: var
                        .attrs
                        .iter()
                        .map(|(name, value)| (name.clone(), attr(value)))
                        .collect(),
                    values: array(file.read_var(&var.name).unwrap()),
                })
                .unwrap();
            }
            let written = nc.finish().unwrap();
            let back = Nc3File::open(&written).unwrap();
            assert_eq!(back.dims, file.dims, "{id}");
            assert_eq!(back.gattrs, file.gattrs, "{id}");
            assert_eq!(back.vars.len(), file.vars.len(), "{id}");
            for var in &vars {
                let read = &back.vars[&var.name];
                assert_eq!(read.dim_ids, var.dim_ids, "{id} {}", var.name);
                for (name, value) in &var.attrs {
                    assert_eq!(
                        bytes(&attr(read.attrs.get(name).unwrap())),
                        bytes(&attr(value)),
                        "{id} {}:{name}",
                        var.name
                    );
                }
                assert_eq!(
                    bytes(&array(back.read_var(&var.name).unwrap())),
                    bytes(&array(file.read_var(&var.name).unwrap())),
                    "{id} {}",
                    var.name
                );
            }
        }
    }

    #[test]
    fn names_are_checked_and_sanitized() {
        let mut nc = Nc3Writer::new();
        assert!(nc.add_dim("time", 0).is_err());
        assert!(nc.add_attr("a/b", Nc3Values::text("x")).is_err());
        assert_eq!(sanitize_name("a/b"), "a_b");
        assert_eq!(sanitize_name(".x"), "_.x");
        assert_eq!(sanitize_name("how.beamwH"), "how.beamwH");
    }
}
