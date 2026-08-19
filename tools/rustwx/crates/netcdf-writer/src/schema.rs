//! The schema: dimensions, global attributes, variables and their
//! attributes, built up before the file is created.
//!
//! Every refusal here fires at DEFINITION time rather than at write time,
//! because the alternative is a half-written file on disk whose header
//! already promised something the data cannot deliver.

use crate::error::{NcWriteError, Result};
use crate::types::{AttrValue, NcFormat, NcType};

/// A dimension. Length 0 marks the record (unlimited) dimension.
#[derive(Debug, Clone)]
pub struct Dim {
    pub name: String,
    pub len: usize,
    pub unlimited: bool,
}

/// A named attribute.
#[derive(Debug, Clone)]
pub struct Attr {
    pub name: String,
    pub value: AttrValue,
}

/// A variable definition. `dimids` index the schema's dimension vector,
/// outermost first (row-major: the last dimension varies fastest).
#[derive(Debug, Clone)]
pub struct VarDef {
    pub name: String,
    pub ty: NcType,
    pub dimids: Vec<usize>,
    pub attrs: Vec<Attr>,
}

/// The complete description of a file, minus its data.
#[derive(Debug, Clone)]
pub struct Schema {
    pub(crate) format: NcFormat,
    pub(crate) dims: Vec<Dim>,
    pub(crate) gattrs: Vec<Attr>,
    pub(crate) vars: Vec<VarDef>,
    pub(crate) record_dim: Option<usize>,
}

/// First character must be a letter or underscore; the rest are the
/// NC-safe subset. Rejects, never renames: a writer that silently
/// rewrites `2m temperature` to `_2m_temperature` produces a tape whose
/// variable cannot be found by the name the caller asked for.
pub fn name_is_valid(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '.' | '@' | '-'))
}

fn check_name(kind: &str, name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(NcWriteError::Schema(format!("{kind} name is empty")));
    }
    if !name_is_valid(name) {
        return Err(NcWriteError::Schema(format!(
            "{kind} name '{name}' is not NC-safe \
             (must start [A-Za-z_], rest [A-Za-z0-9_+.@-])"
        )));
    }
    Ok(())
}

impl Schema {
    /// An empty schema for the given container version.
    pub fn new(format: NcFormat) -> Self {
        Schema {
            format,
            dims: Vec::new(),
            gattrs: Vec::new(),
            vars: Vec::new(),
            record_dim: None,
        }
    }

    /// Which container version this schema targets.
    pub fn format(&self) -> NcFormat {
        self.format
    }

    /// Define a dimension; returns its dimid.
    ///
    /// `unlimited` marks the record dimension, of which the classic
    /// format allows exactly one. `len` is ignored when `unlimited` is
    /// set -- the record count comes from the data.
    pub fn def_dim(&mut self, name: &str, len: usize, unlimited: bool) -> Result<usize> {
        check_name("dimension", name)?;
        if self.dims.iter().any(|d| d.name == name) {
            return Err(NcWriteError::Schema(format!(
                "duplicate dimension name '{name}'"
            )));
        }
        if unlimited {
            if let Some(existing) = self.record_dim {
                return Err(NcWriteError::Schema(format!(
                    "'{name}' would be a second unlimited dimension; the classic \
                     format allows exactly one and '{}' already claims it",
                    self.dims[existing].name
                )));
            }
        } else if len == 0 {
            return Err(NcWriteError::Schema(format!(
                "dimension '{name}' has length 0; only the unlimited dimension \
                 may be zero-length (pass unlimited=true if that is what it is)"
            )));
        }
        let dimid = self.dims.len();
        if unlimited {
            self.record_dim = Some(dimid);
        }
        self.dims.push(Dim {
            name: name.to_string(),
            len: if unlimited { 0 } else { len },
            unlimited,
        });
        Ok(dimid)
    }

    /// Define a variable; returns its varid.
    pub fn def_var(&mut self, name: &str, ty: NcType, dimids: &[usize]) -> Result<usize> {
        check_name("variable", name)?;
        if self.vars.iter().any(|v| v.name == name) {
            return Err(NcWriteError::Schema(format!(
                "duplicate variable name '{name}'"
            )));
        }
        if ty.is_cdf5_only() && !matches!(self.format, NcFormat::Cdf5) {
            return Err(NcWriteError::Schema(format!(
                "variable '{name}' is {} which exists only in CDF-5; this file is \
                 {} and a {} reader has no code for it. Create the schema with \
                 NcFormat::Cdf5, or narrow the type",
                ty.name(),
                self.format.label(),
                self.format.label(),
            )));
        }
        for (position, &dimid) in dimids.iter().enumerate() {
            let dim = self.dims.get(dimid).ok_or_else(|| {
                NcWriteError::Schema(format!(
                    "variable '{name}' references dimid {dimid} but only {} \
                     dimension(s) are defined",
                    self.dims.len()
                ))
            })?;
            if dim.unlimited && position != 0 {
                return Err(NcWriteError::Schema(format!(
                    "variable '{name}' puts the record dimension '{}' at position \
                     {position}; classic NetCDF interleaves records, so the record \
                     dimension must be the first dimension of a record variable",
                    dim.name
                )));
            }
        }
        let mut seen = Vec::with_capacity(dimids.len());
        for &dimid in dimids {
            if seen.contains(&dimid) {
                return Err(NcWriteError::Schema(format!(
                    "variable '{name}' uses dimension '{}' twice",
                    self.dims[dimid].name
                )));
            }
            seen.push(dimid);
        }
        self.vars.push(VarDef {
            name: name.to_string(),
            ty,
            dimids: dimids.to_vec(),
            attrs: Vec::new(),
        });
        Ok(self.vars.len() - 1)
    }

    /// Attach a global attribute.
    pub fn put_global_attr(&mut self, name: &str, value: AttrValue) -> Result<()> {
        check_name("attribute", name)?;
        self.check_attr_type("global attribute", name, &value)?;
        if self.gattrs.iter().any(|a| a.name == name) {
            return Err(NcWriteError::Schema(format!(
                "duplicate global attribute name '{name}'"
            )));
        }
        self.gattrs.push(Attr {
            name: name.to_string(),
            value,
        });
        Ok(())
    }

    /// Attach an attribute to a variable.
    pub fn put_var_attr(&mut self, varid: usize, name: &str, value: AttrValue) -> Result<()> {
        check_name("attribute", name)?;
        self.check_attr_type("variable attribute", name, &value)?;
        let nvars = self.vars.len();
        let var = self
            .vars
            .get_mut(varid)
            .ok_or_else(|| NcWriteError::Schema(format!(
                "attribute '{name}' targets varid {varid} but only {nvars} variable(s) are defined"
            )))?;
        if var.attrs.iter().any(|a| a.name == name) {
            return Err(NcWriteError::Schema(format!(
                "duplicate attribute name '{name}' on variable '{}'",
                var.name
            )));
        }
        var.attrs.push(Attr {
            name: name.to_string(),
            value,
        });
        Ok(())
    }

    fn check_attr_type(&self, scope: &str, name: &str, value: &AttrValue) -> Result<()> {
        let ty = value.nc_type();
        if ty.is_cdf5_only() && !matches!(self.format, NcFormat::Cdf5) {
            return Err(NcWriteError::Schema(format!(
                "{scope} '{name}' is {} which exists only in CDF-5; this file is {}",
                ty.name(),
                self.format.label()
            )));
        }
        Ok(())
    }

    /// The dimid of the record dimension, if one was defined.
    pub fn record_dim(&self) -> Option<usize> {
        self.record_dim
    }

    /// Does this variable ride the record dimension?
    pub(crate) fn is_record_var(&self, varid: usize) -> bool {
        match self.record_dim {
            None => false,
            Some(rec) => self.vars[varid].dimids.first() == Some(&rec),
        }
    }

    /// Number of variables defined.
    pub fn num_vars(&self) -> usize {
        self.vars.len()
    }

    /// Number of dimensions defined.
    pub fn num_dims(&self) -> usize {
        self.dims.len()
    }

    /// A variable's name, for callers that only kept the varid.
    pub fn var_name(&self, varid: usize) -> Option<&str> {
        self.vars.get(varid).map(|v| v.name.as_str())
    }
}
