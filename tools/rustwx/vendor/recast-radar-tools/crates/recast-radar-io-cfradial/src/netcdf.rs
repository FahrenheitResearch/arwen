//! One netCDF view over a classic file or one group of a netCDF-4 file.
//!
//! The CfRadial decoders read dimensions, attributes and variables through
//! [`NcFile`] whichever container holds them: a classic (CDF-1/CDF-2) file
//! parsed by [`crate::netcdf3`], or a group of a netCDF-4 file read through
//! [`recast_radar_hdf5::netcdf4`]. A group view sees the dimensions of the
//! group and of its ancestors (the group's own first, so an inner dimension
//! shadows an outer one of the same name, as in netCDF-C), the group's
//! variables and its attributes, in file order. Values keep their stored
//! type ([`NcArray`]). netCDF-4 user-defined types are spread over the
//! netCDF types: a compound variable or attribute becomes one per member,
//! named `<name>.<member>` (nested members `<name>.<member>.<member>`); an
//! enumeration keeps its base integers; an opaque attribute becomes its
//! bytes (`ubyte`), an opaque variable its bytes over one more dimension,
//! `<name>.bytes`; a variable-length attribute its values back to back with
//! `<name>.lengths`, a variable-length variable `<name>.lengths` over its
//! dimensions and its values back to back in `<name>` over a dimension
//! `<name>.values`.

use std::collections::BTreeMap;

use recast_radar_hdf5::netcdf4::{NcFile as Nc4File, NcType};
use recast_radar_hdf5::{Datatype, Values};

use crate::netcdf3::{IntKind, Location, Nc3File, NcArray, NcAttrs, NcValue, NcVar};
use crate::{CfRadialError, Result};

/// A netCDF dimension table, attribute map and variable map with a reader
/// for the variables' data.
pub struct NcFile<'f> {
    /// `(name, length)` of every visible dimension; `NcVar::dim_ids` index
    /// it.
    pub dims: Vec<(String, usize)>,
    /// Global (classic) or group (netCDF-4) attributes, in file order.
    pub gattrs: NcAttrs,
    /// Variables by name.
    pub vars: BTreeMap<String, NcVar>,
    backend: Backend<'f>,
}

enum Backend<'f> {
    Classic(Nc3File<'f>),
    Netcdf4(&'f Nc4File<'f>),
}

impl<'f> NcFile<'f> {
    /// The view of a classic netCDF file.
    pub fn classic(file: Nc3File<'f>) -> Self {
        Self {
            dims: file.dims.clone(),
            gattrs: file.gattrs.clone(),
            vars: file.vars.clone(),
            backend: Backend::Classic(file),
        }
    }

    /// The view of the netCDF-4 group at `path`.
    pub fn netcdf4_group(file: &'f Nc4File<'f>, path: &str) -> Result<Self> {
        let group = file
            .group(path)
            .ok_or_else(|| CfRadialError::InvalidMessage {
                offset: 0,
                reason: format!("netCDF-4 group '{path}' not found"),
            })?;
        let visible = file.visible_dims(path);
        let mut dims: Vec<(String, usize)> = visible
            .iter()
            .filter_map(|id| file.dim(*id).map(|dim| (dim.name.clone(), dim.len)))
            .collect();
        let mut vars = BTreeMap::new();
        let mut index = 0usize;
        for variable in &group.variables {
            let dim_ids = variable
                .dims
                .iter()
                .map(|id| {
                    visible
                        .iter()
                        .position(|visible| visible == id)
                        .ok_or_else(|| CfRadialError::InvalidMessage {
                            offset: 0,
                            reason: format!(
                                "netCDF-4 variable '{}' uses a dimension outside its group's scope",
                                variable.path
                            ),
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let attrs = attributes(&variable.attributes);
            // An opaque variable: its bytes, over one more dimension of the
            // opaque size.
            if let Datatype::Opaque { size, .. } = &variable.datatype {
                let mut dim_ids = dim_ids;
                dim_ids.push(dims.len());
                dims.push((format!("{}.bytes", variable.name), *size));
                vars.insert(
                    variable.name.clone(),
                    NcVar {
                        name: variable.name.clone(),
                        dim_ids,
                        attrs,
                        index,
                        nc_type: type_code(NcType::UByte),
                        location: Location::Hdf5Bytes(variable.path.clone()),
                    },
                );
                index += 1;
                continue;
            }
            // A variable-length variable: `<name>.lengths` over its
            // dimensions and the values back to back in `<name>` over a
            // dimension of their count (read now, to size it; one that does
            // not read has no values variable).
            if let Datatype::VarLenSequence { base, .. } = &variable.datatype {
                let values = file
                    .hdf5()
                    .dataset(&variable.path)
                    .ok()
                    .and_then(|dataset| vlen_parts(dataset.values, &variable.name).ok());
                vars.insert(
                    format!("{}.lengths", variable.name),
                    NcVar {
                        name: format!("{}.lengths", variable.name),
                        dim_ids,
                        attrs: attrs.clone(),
                        index,
                        nc_type: type_code(NcType::Int64),
                        location: Location::Hdf5VlenLengths(variable.path.clone()),
                    },
                );
                index += 1;
                if let Some((_, flat)) = values {
                    let dim = dims.len();
                    dims.push((format!("{}.values", variable.name), flat.len()));
                    vars.insert(
                        variable.name.clone(),
                        NcVar {
                            name: variable.name.clone(),
                            dim_ids: vec![dim],
                            attrs,
                            index,
                            nc_type: type_code(NcType::of(base)),
                            location: Location::Hdf5VlenValues(variable.path.clone()),
                        },
                    );
                    index += 1;
                }
                continue;
            }
            // A compound variable: one variable per member.
            let mut members = Vec::new();
            compound_members(&variable.datatype, &mut Vec::new(), &mut members, 0);
            if members.is_empty() {
                vars.insert(
                    variable.name.clone(),
                    NcVar {
                        name: variable.name.clone(),
                        dim_ids,
                        attrs,
                        index,
                        nc_type: type_code(variable.nc_type),
                        location: Location::Hdf5(variable.path.clone()),
                    },
                );
                index += 1;
                continue;
            }
            for (path, datatype) in members {
                let name = format!("{}.{}", variable.name, path.join("."));
                vars.insert(
                    name.clone(),
                    NcVar {
                        name,
                        dim_ids: dim_ids.clone(),
                        attrs: attrs.clone(),
                        index,
                        nc_type: type_code(NcType::of(&datatype)),
                        location: Location::Hdf5Member {
                            path: variable.path.clone(),
                            members: path,
                        },
                    },
                );
                index += 1;
            }
        }
        Ok(Self {
            dims,
            gattrs: attributes(&group.attributes),
            vars,
            backend: Backend::Netcdf4(file),
        })
    }

    /// True for a view of a netCDF-4 group.
    pub fn is_netcdf4(&self) -> bool {
        matches!(self.backend, Backend::Netcdf4(_))
    }

    /// A text attribute of the file or group.
    pub fn gattr_str(&self, name: &str) -> Option<&str> {
        self.gattrs.get(name).and_then(NcValue::as_str)
    }

    /// The first element of a numeric attribute of the file or group.
    pub fn gattr_f64(&self, name: &str) -> Option<f64> {
        self.gattrs.get(name).and_then(NcValue::as_f64)
    }

    /// Dimension lengths of a variable.
    pub fn var_dims(&self, var: &NcVar) -> Vec<usize> {
        var.dim_ids
            .iter()
            .map(|id| self.dims.get(*id).map(|(_, len)| *len).unwrap_or(0))
            .collect()
    }

    /// Name of a variable's dimension `axis`.
    pub fn dim_name(&self, var: &NcVar, axis: usize) -> Option<&str> {
        let id = *var.dim_ids.get(axis)?;
        self.dims.get(id).map(|(name, _)| name.as_str())
    }

    /// Read the full data array of `name`.
    pub fn read_var(&self, name: &str) -> Result<NcArray> {
        match &self.backend {
            Backend::Classic(file) => file.read_var(name),
            Backend::Netcdf4(file) => {
                let var = self
                    .vars
                    .get(name)
                    .ok_or_else(|| CfRadialError::InvalidMessage {
                        offset: 0,
                        reason: format!("netCDF variable '{name}' not found"),
                    })?;
                match &var.location {
                    Location::Hdf5(path) => array(file.hdf5().dataset(path)?.values, name),
                    Location::Hdf5Member { path, members } => {
                        let mut values = file.hdf5().dataset(path)?.values;
                        for member in members {
                            let Values::Compound {
                                members: columns, ..
                            } = values
                            else {
                                return Err(CfRadialError::InvalidMessage {
                                    offset: 0,
                                    reason: format!("netCDF variable '{name}' is not a compound"),
                                });
                            };
                            values = columns
                                .into_iter()
                                .find(|(have, _)| have == member)
                                .map(|(_, column)| column)
                                .ok_or_else(|| CfRadialError::InvalidMessage {
                                    offset: 0,
                                    reason: format!(
                                        "netCDF variable '{name}' has no member {member}"
                                    ),
                                })?;
                        }
                        array(values, name)
                    }
                    Location::Hdf5Bytes(path) => match file.hdf5().dataset(path)?.values {
                        Values::Raw { bytes, .. } => Ok(NcArray::U8(bytes)),
                        _ => Err(CfRadialError::InvalidMessage {
                            offset: 0,
                            reason: format!("netCDF variable '{name}' is not opaque"),
                        }),
                    },
                    Location::Hdf5VlenLengths(path) => {
                        vlen_parts(file.hdf5().dataset(path)?.values, name)
                            .map(|(lengths, _)| lengths)
                    }
                    Location::Hdf5VlenValues(path) => {
                        vlen_parts(file.hdf5().dataset(path)?.values, name).map(|(_, flat)| flat)
                    }
                    Location::Classic(_) => Err(CfRadialError::InvalidMessage {
                        offset: 0,
                        reason: format!("netCDF variable '{name}' is not in a netCDF-4 file"),
                    }),
                }
            }
        }
    }
}

/// A variable-length variable's sequence lengths (`int64`) and its values
/// back to back, which must all be of one numeric type.
fn vlen_parts(values: Values, name: &str) -> Result<(NcArray, NcArray)> {
    let not_numeric = || CfRadialError::InvalidMessage {
        offset: 0,
        reason: format!("netCDF variable '{name}' is not a variable-length numeric variable"),
    };
    let Values::Sequences(sequences) = values else {
        return Err(not_numeric());
    };
    let lengths = NcArray::I64(sequences.iter().map(|s| s.len() as i64).collect());
    let mut flat: Option<NcArray> = None;
    for sequence in sequences {
        let part = array(sequence, name)?;
        flat = Some(match flat {
            None => part,
            Some(mut joined) => {
                macro_rules! join {
                    ($($variant:ident),*) => {
                        match (&mut joined, part) {
                            $((NcArray::$variant(a), NcArray::$variant(b)) => a.extend(b),)*
                            _ => return Err(not_numeric()),
                        }
                    };
                }
                join!(I8, U8, I16, U16, I32, U32, I64, U64, F32, F64);
                joined
            }
        });
    }
    // An empty variable joins to no values; its base type is unknown here.
    Ok((lengths, flat.unwrap_or(NcArray::F64(Vec::new()))))
}

/// netCDF type codes (netCDF-C `nc_type`); 0 for a user-defined type.
fn type_code(nc_type: NcType) -> u32 {
    match nc_type {
        NcType::Byte => 1,
        NcType::Char => 2,
        NcType::Short => 3,
        NcType::Int => 4,
        NcType::Float => 5,
        NcType::Double => 6,
        NcType::UByte => 7,
        NcType::UShort => 8,
        NcType::UInt => 9,
        NcType::Int64 => 10,
        NcType::UInt64 => 11,
        NcType::String => 12,
        _ => 0,
    }
}

/// The members of a compound datatype that are not compounds themselves,
/// with their member-name paths (empty for any other datatype).
fn compound_members(
    datatype: &Datatype,
    path: &mut Vec<String>,
    out: &mut Vec<(Vec<String>, Datatype)>,
    depth: usize,
) {
    let Datatype::Compound { members, .. } = datatype else {
        return;
    };
    // Datatype nesting is bounded by the HDF5 reader (16).
    if depth > 16 {
        return;
    }
    for member in members {
        path.push(member.name.clone());
        if matches!(member.datatype, Datatype::Compound { .. }) {
            compound_members(&member.datatype, path, out, depth + 1);
        } else {
            out.push((path.clone(), member.datatype.clone()));
        }
        path.pop();
    }
}

fn attributes(list: &[recast_radar_hdf5::Attribute]) -> NcAttrs {
    let mut out = NcAttrs::default();
    for attribute in list {
        if attribute.is_null() {
            // netCDF-C writes an empty text attribute as a null dataspace.
            if attribute.datatype().is_string() {
                out.insert(attribute.name().to_owned(), NcValue::Str(String::new()));
            }
            continue;
        }
        spread(attribute.name(), attribute.values(), &mut out, 0);
    }
    out
}

/// An attribute's values as netCDF attributes: itself, or one per compound
/// member (`<name>.<member>`), an opaque value as its bytes, a
/// variable-length value's elements back to back with `<name>.lengths`.
fn spread(name: &str, values: &Values, out: &mut NcAttrs, depth: usize) {
    if let Some(value) = value(values) {
        out.insert(name.to_owned(), value);
        return;
    }
    if depth > 16 {
        return;
    }
    match values {
        Values::Compound {
            members: columns, ..
        } => {
            for (member, column) in columns {
                spread(&format!("{name}.{member}"), column, out, depth + 1);
            }
        }
        Values::Raw { bytes, .. } => {
            out.insert(
                name.to_owned(),
                NcValue::Ints(bytes.iter().map(|b| i64::from(*b)).collect(), IntKind::U8),
            );
        }
        Values::Sequences(sequences) => {
            let lengths: Vec<i64> = sequences.iter().map(|s| s.len() as i64).collect();
            let joined: Vec<NcValue> = sequences.iter().filter_map(value).collect();
            if joined.len() == sequences.len()
                && let Some(value) = concat_values(joined.iter())
            {
                out.insert(name.to_owned(), value);
                out.insert(
                    format!("{name}.lengths"),
                    NcValue::Ints(lengths, IntKind::I64),
                );
            }
        }
        _ => {}
    }
}

/// Numeric values of one kind joined (`None` for mixed kinds or text).
fn concat_values<'a>(mut values: impl Iterator<Item = &'a NcValue>) -> Option<NcValue> {
    let mut out = values.next()?.clone();
    for value in values {
        match (&mut out, value) {
            (NcValue::Floats(a), NcValue::Floats(b)) => a.extend_from_slice(b),
            (NcValue::Doubles(a), NcValue::Doubles(b)) => a.extend_from_slice(b),
            (NcValue::Ints(a, kind), NcValue::Ints(b, other)) if kind == other => {
                a.extend_from_slice(b);
            }
            _ => return None,
        }
    }
    Some(out)
}

fn value(values: &Values) -> Option<NcValue> {
    if let Some(mut strings) = values.strings() {
        return Some(if strings.len() == 1 {
            NcValue::Str(strings.remove(0))
        } else {
            NcValue::Strings(strings)
        });
    }
    Some(match values {
        Values::F32(v) => NcValue::Floats(v.clone()),
        Values::F64(v) => NcValue::Doubles(v.clone()),
        Values::U64(v) if v.iter().any(|x| i64::try_from(*x).is_err()) => {
            NcValue::Doubles(v.iter().map(|x| *x as f64).collect())
        }
        values if values.is_numeric() => {
            let kind = match values {
                Values::I8(_) => IntKind::I8,
                Values::U8(_) => IntKind::U8,
                Values::I16(_) => IntKind::I16,
                Values::U16(_) => IntKind::U16,
                Values::I32(_) => IntKind::I32,
                Values::U32(_) => IntKind::U32,
                Values::U64(_) => IntKind::U64,
                _ => IntKind::I64,
            };
            NcValue::Ints(
                (0..values.len())
                    .filter_map(|i| values.get_i64(i))
                    .collect(),
                kind,
            )
        }
        _ => return None,
    })
}

fn array(values: Values, name: &str) -> Result<NcArray> {
    Ok(match values {
        Values::I8(v) => NcArray::I8(v),
        Values::U8(v) => NcArray::U8(v),
        Values::I16(v) => NcArray::I16(v),
        Values::U16(v) => NcArray::U16(v),
        Values::I32(v) => NcArray::I32(v),
        Values::U32(v) => NcArray::U32(v),
        Values::I64(v) => NcArray::I64(v),
        Values::U64(v) => NcArray::U64(v),
        Values::F32(v) => NcArray::F32(v),
        Values::F64(v) => NcArray::F64(v),
        Values::FixedStrings { size: 1, bytes, .. } => NcArray::Char(bytes),
        other => match other.strings() {
            Some(strings) => NcArray::Str(strings),
            None => {
                return Err(CfRadialError::InvalidMessage {
                    offset: 0,
                    reason: format!("netCDF-4 variable '{name}' has a user-defined type"),
                });
            }
        },
    })
}
