//! CfRadial 2 / WMO FM301 writer: the FM301 view of a volume
//! ([`recast_radar_core::fm301::volume_view`], FM301-2022 flavor, rays in
//! time order) written as netCDF-4 with
//! [`recast_radar_hdf5::write::netcdf4::NcWriter`].
//!
//! Every group, dimension, variable and attribute of the view is written as
//! the view states it: the root (global attributes, `volume_number`,
//! `time_coverage_*`, the location, `sweep_group_name`,
//! `sweep_fixed_angle`), `radar_parameters`, `radar_calibration`,
//! `georeferencing_correction`, and one `sweep_<n>` group per sweep with
//! its coordinates, instrument variables, `monitoring` subgroup and fields.
//! Fields keep their storage type and raw codes with the packing attributes
//! in the packed type; they are chunked by rays, byte-shuffled and deflated.
//! Text variables are netCDF-4 `string`s, text attributes `char`, bool
//! attributes `"true"`/`"false"`.
//!
//! By default the view keeps everything the volume holds
//! ([`Passthrough::All`]): source attributes and variables without an
//! FM301 name are written verbatim. Attributes netCDF-4 reserves
//! (`_NCProperties`, `CLASS`, `NAME`, `DIMENSION_LIST`, `REFERENCE_LIST`,
//! `_Netcdf4Dimid`, `_Netcdf4Coordinates`, `_nc3_strict`) describe the
//! source container and are not written; names netCDF cannot store are made
//! valid (`/` and control characters become `_`).
//!
//! A sweep of more than 16,384 gates per ray, or fields beyond the 1 GiB
//! decode budget, is [`CfWriteError::TooLarge`] before anything is
//! allocated: that is what this crate's readers accept, so every file
//! written reads back.
//!
//! An integer field with rows the source did not provide, or gates the
//! sweep's range has beyond its own, needs a `_FillValue` for them. When its
//! coding has none, the writer takes a code no gate of the field uses (as
//! the CfRadial 1 writer does) and writes those rows and gates with it; a
//! field that uses every code of its type is
//! [`CfWriteError::Unrepresentable`].

use std::borrow::Cow;
use std::collections::HashSet;

use recast_radar_core::bounded_read::{MAX_DECODED_VOLUME_BYTES, MAX_GATES_PER_RADIAL};
use recast_radar_core::fm301::{
    self, ArrayRef, FirstDim, Flavor, Group, Passthrough, Values, Variable, ViewOptions,
};
use recast_radar_core::model::{
    ArrayBuf, AttrValue, Field, FieldData, IntCoding, Scalar, Sweep, Volume,
};
use recast_radar_hdf5::write::netcdf4::{GroupId, NcAttr, NcStorage, NcVariable, NcWriter};
use recast_radar_hdf5::write::{CharSet, Data};

use super::CfWriteError;
use super::cfradial1::Code;
use super::netcdf3::sanitize_name;

/// Attributes netCDF-4 reserves.
const RESERVED: &[&str] = &[
    "CLASS",
    "NAME",
    "REFERENCE_LIST",
    "DIMENSION_LIST",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "_nc3_strict",
    "_NCProperties",
];

/// Options of [`write_cfradial2`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Cfradial2Options {
    /// zlib level of the fields (0-9); `None` stores them uncompressed.
    /// Default 4.
    pub deflate: Option<u32>,
    /// Write every source item without an FM301 name too
    /// ([`Passthrough::All`]); `false` writes FM301 names only. Default
    /// `true`.
    pub passthrough: bool,
}

impl Default for Cfradial2Options {
    fn default() -> Self {
        Self {
            deflate: Some(4),
            passthrough: true,
        }
    }
}

impl Cfradial2Options {
    /// The same options with this field compression.
    pub fn with_deflate(mut self, deflate: Option<u32>) -> Self {
        self.deflate = deflate;
        self
    }

    /// The same options, writing source items without an FM301 name or
    /// not.
    pub fn with_passthrough(mut self, passthrough: bool) -> Self {
        self.passthrough = passthrough;
        self
    }
}

/// Write `volume` as CfRadial 2 / FM301 netCDF-4. See the module
/// documentation.
pub fn write_cfradial2(
    volume: &Volume,
    options: &Cfradial2Options,
) -> Result<Vec<u8>, CfWriteError> {
    if volume.sweeps.is_empty() {
        return Err(CfWriteError::Unrepresentable(
            "a volume without sweeps".into(),
        ));
    }
    check_size(volume)?;
    let volume = with_fill_codes(volume)?;
    let view = fm301::volume_view(
        &volume,
        ViewOptions {
            flavor: Flavor::Wmo2022,
            first_dim: FirstDim::Time,
            passthrough: if options.passthrough {
                Passthrough::All
            } else {
                Passthrough::Flavor
            },
        },
        None,
    )?;
    let mut nc = NcWriter::new();
    let root = nc.root();
    write_group(&mut nc, root, &view.root, options)?;
    Ok(nc.finish()?)
}

/// Refuse, before the view materialises anything, what no reader of this
/// crate would read back: a sweep of more than [`MAX_GATES_PER_RADIAL`] gates
/// per ray (a range coordinate is three numbers in the model, but the file
/// states every centre and every field row), or fields beyond a reader's
/// decode budget ([`MAX_DECODED_VOLUME_BYTES`]).
fn check_size(volume: &Volume) -> Result<(), CfWriteError> {
    let mut bytes = 0usize;
    for (index, sweep) in volume.sweeps.iter().enumerate() {
        let ngates = sweep.range.ngates();
        if ngates > MAX_GATES_PER_RADIAL {
            return Err(CfWriteError::TooLarge(format!(
                "sweep {index}: {ngates} gates per ray (at most {MAX_GATES_PER_RADIAL})"
            )));
        }
        let cells = sweep.nrays().saturating_mul(ngates);
        for field in &sweep.fields {
            let element = match &field.data {
                FieldData::U8 { .. } | FieldData::I8 { .. } => 1,
                FieldData::U16 { .. } | FieldData::I16 { .. } => 2,
                FieldData::I32 { .. } | FieldData::F32 { .. } => 4,
                FieldData::F64 { .. } => 8,
            };
            bytes = bytes.saturating_add(cells.saturating_mul(element));
        }
    }
    if bytes > MAX_DECODED_VOLUME_BYTES {
        return Err(CfWriteError::TooLarge(format!(
            "{bytes} bytes of fields (at most {MAX_DECODED_VOLUME_BYTES})"
        )));
    }
    Ok(())
}

/// `true` when `field` has gates the file must write as fill (rows the
/// source did not provide, range gates before or after its own) and an
/// integer coding without a fill code.
fn lacks_fill(sweep: &Sweep, field: &Field) -> bool {
    let no_fill = match &field.data {
        FieldData::U8 { coding, .. } => coding.fill_value.is_none(),
        FieldData::U16 { coding, .. } => coding.fill_value.is_none(),
        FieldData::I8 { coding, .. } => coding.fill_value.is_none(),
        FieldData::I16 { coding, .. } => coding.fill_value.is_none(),
        FieldData::I32 { coding, .. } => coding.fill_value.is_none(),
        FieldData::F32 { .. } | FieldData::F64 { .. } => false,
    };
    let covered = u64::from(field.gates.start).saturating_add(
        u64::from(field.ngates).saturating_mul(u64::from(field.gates.stride.max(1))),
    );
    no_fill
        && (!field.absent_rows.is_empty()
            || field.gates.start > 0
            || covered < sweep.range.ngates() as u64
            || (field.nrays as usize) < sweep.nrays())
}

/// `volume`, with a fill code for every integer field that needs one and
/// has none ([`lacks_fill`]): a code none of its provided gates uses, also
/// written into its absent rows. Borrowed when no field needs one.
fn with_fill_codes(volume: &Volume) -> Result<Cow<'_, Volume>, CfWriteError> {
    let needs = |sweep: &Sweep| sweep.fields.iter().any(|field| lacks_fill(sweep, field));
    if !volume.sweeps.iter().any(needs) {
        return Ok(Cow::Borrowed(volume));
    }
    let mut owned = volume.clone();
    for sweep in &mut owned.sweeps {
        let lacking: Vec<bool> = sweep
            .fields
            .iter()
            .map(|field| lacks_fill(sweep, field))
            .collect();
        for (field, lacking) in sweep.fields.iter_mut().zip(lacking) {
            if !lacking {
                continue;
            }
            let Field {
                name,
                ngates,
                data,
                absent_rows,
                ..
            } = field;
            let name = name.as_str();
            let ngates = *ngates as usize;
            match data {
                FieldData::U8 { values, coding } => {
                    state_fill(values, coding, absent_rows, ngates, name)?;
                }
                FieldData::U16 { values, coding } => {
                    state_fill(values, coding, absent_rows, ngates, name)?;
                }
                FieldData::I8 { values, coding } => {
                    state_fill(values, coding, absent_rows, ngates, name)?;
                }
                FieldData::I16 { values, coding } => {
                    state_fill(values, coding, absent_rows, ngates, name)?;
                }
                FieldData::I32 { values, coding } => {
                    state_fill(values, coding, absent_rows, ngates, name)?;
                }
                FieldData::F32 { .. } | FieldData::F64 { .. } => {}
            }
        }
    }
    Ok(Cow::Owned(owned))
}

/// Give `coding` a fill code no provided row of `values` uses (nor its
/// undetect or range-folded code), and write it into the absent rows.
fn state_fill<T: Code>(
    values: &mut [T],
    coding: &mut IntCoding<T>,
    absent_rows: &[u32],
    ngates: usize,
    name: &str,
) -> Result<(), CfWriteError> {
    let absent: HashSet<u32> = absent_rows.iter().copied().collect();
    let used: HashSet<T> = values
        .chunks(ngates.max(1))
        .enumerate()
        .filter(|(row, _)| !absent.contains(&(*row as u32)))
        .flat_map(|(_, row)| row.iter().copied())
        .collect();
    let taken = [coding.undetect, coding.range_folded];
    let fill = T::candidates()
        .find(|code| !used.contains(code) && !taken.contains(&Some(*code)))
        .ok_or_else(|| {
            CfWriteError::Unrepresentable(format!(
                "field {name}: no free code for _FillValue (rows or gates without values)"
            ))
        })?;
    coding.fill_value = Some(fill);
    for row in absent_rows {
        let start = (*row as usize).saturating_mul(ngates);
        if let Some(cells) = values.get_mut(start..start.saturating_add(ngates)) {
            cells.fill(fill);
        }
    }
    Ok(())
}

fn write_group(
    nc: &mut NcWriter,
    id: GroupId,
    group: &Group<'_>,
    options: &Cfradial2Options,
) -> Result<(), CfWriteError> {
    for (name, len) in &group.dims {
        nc.add_dim(id, &sanitize_name(name), *len as u64)?;
    }
    let mut seen = Vec::new();
    for (name, value) in &group.attrs {
        let name = sanitize_name(name);
        if RESERVED.contains(&name.as_str()) || seen.contains(&name) {
            continue;
        }
        if let Some(value) = attr(value) {
            nc.add_attr(id, &name, value)?;
            seen.push(name);
        }
    }
    for variable in &group.variables {
        nc.add_variable(id, nc_variable(variable, &group.dims, options)?)?;
    }
    for child in &group.children {
        let child_id = nc.add_group(id, &sanitize_name(&child.name))?;
        write_group(nc, child_id, child, options)?;
    }
    Ok(())
}

/// A view attribute as a netCDF-4 attribute (`None` for an empty array).
fn attr(value: &AttrValue) -> Option<NcAttr> {
    Some(match value {
        AttrValue::Text(text) => NcAttr::Text(text.to_string()),
        AttrValue::Bool(value) => NcAttr::Text(if *value { "true" } else { "false" }.into()),
        AttrValue::Scalar(scalar) => NcAttr::Numbers(scalar_data(*scalar)),
        AttrValue::Array(ArrayBuf::Text(texts)) => {
            NcAttr::Strings(texts.iter().map(|t| t.to_string()).collect())
        }
        AttrValue::Array(array) => {
            if array.is_empty() {
                return None;
            }
            NcAttr::Numbers(array_data(array))
        }
    })
}

fn scalar_data(scalar: Scalar) -> Data {
    match scalar {
        Scalar::I8(v) => Data::I8(vec![v]),
        Scalar::U8(v) => Data::U8(vec![v]),
        Scalar::I16(v) => Data::I16(vec![v]),
        Scalar::U16(v) => Data::U16(vec![v]),
        Scalar::I32(v) => Data::I32(vec![v]),
        Scalar::U32(v) => Data::U32(vec![v]),
        Scalar::I64(v) => Data::I64(vec![v]),
        Scalar::U64(v) => Data::U64(vec![v]),
        Scalar::F32(v) => Data::F32(vec![v]),
        Scalar::F64(v) => Data::F64(vec![v]),
    }
}

fn array_data(array: &ArrayBuf) -> Data {
    match array {
        ArrayBuf::I8(v) => Data::I8(v.clone()),
        ArrayBuf::U8(v) => Data::U8(v.clone()),
        ArrayBuf::I16(v) => Data::I16(v.clone()),
        ArrayBuf::U16(v) => Data::U16(v.clone()),
        ArrayBuf::I32(v) => Data::I32(v.clone()),
        ArrayBuf::U32(v) => Data::U32(v.clone()),
        ArrayBuf::I64(v) => Data::I64(v.clone()),
        ArrayBuf::F32(v) => Data::F32(v.clone()),
        ArrayBuf::F64(v) => Data::F64(v.clone()),
        ArrayBuf::Text(v) => Data::VarStrings {
            values: v.iter().map(|t| t.to_string()).collect(),
            charset: CharSet::Utf8,
        },
    }
}

fn ref_data(array: ArrayRef<'_>) -> Data {
    match array {
        ArrayRef::U8(v) => Data::U8(v.to_vec()),
        ArrayRef::U16(v) => Data::U16(v.to_vec()),
        ArrayRef::I8(v) => Data::I8(v.to_vec()),
        ArrayRef::I16(v) => Data::I16(v.to_vec()),
        ArrayRef::I32(v) => Data::I32(v.to_vec()),
        ArrayRef::F32(v) => Data::F32(v.to_vec()),
        ArrayRef::F64(v) => Data::F64(v.to_vec()),
    }
}

/// Rays per chunk of a field: all of them up to 4 MiB, else about 1 MiB
/// chunks.
fn chunk_rows(nrays: usize, ngates: usize, element: usize) -> usize {
    let row = ngates.max(1) * element;
    if nrays * row <= 4 << 20 {
        nrays.max(1)
    } else {
        ((1 << 20) / row).clamp(1, nrays.max(1))
    }
}

fn element_size(data: &Data) -> usize {
    match data {
        Data::I8(_) | Data::U8(_) => 1,
        Data::I16(_) | Data::U16(_) => 2,
        Data::I32(_) | Data::U32(_) | Data::F32(_) => 4,
        _ => 8,
    }
}

fn nc_variable(
    variable: &Variable<'_>,
    group_dims: &[(std::borrow::Cow<'_, str>, usize)],
    options: &Cfradial2Options,
) -> Result<NcVariable, CfWriteError> {
    let data = match &variable.values {
        Values::Borrowed(array) => ref_data(*array),
        Values::Owned(array) => array_data(array),
        Values::Mapped { .. } => array_data(&variable.values.materialize().ok_or_else(|| {
            CfWriteError::Invalid(format!("variable {}: no values", variable.name))
        })?),
        Values::Scalar(scalar) => scalar_data(*scalar),
        Values::Text(text) => Data::VarStrings {
            values: vec![text.to_string()],
            charset: CharSet::Utf8,
        },
    };
    let dims: Vec<String> = variable.dims.iter().map(|dim| sanitize_name(dim)).collect();
    // Fields (two-dimensional numeric variables of a model field) are
    // chunked by rays and deflated; everything else is contiguous.
    let storage = match (&variable.source, dims.len(), &data) {
        (Some(_), 2, data) if !matches!(data, Data::VarStrings { .. }) => {
            let len = |dim: &str| {
                group_dims
                    .iter()
                    .find(|(name, _)| name == dim)
                    .map_or(0, |(_, len)| *len)
            };
            let (nrays, ngates) = (len(&variable.dims[0]), len(&variable.dims[1]));
            if options.deflate.is_some() && nrays > 0 && ngates > 0 {
                NcStorage::Chunked {
                    chunk: vec![
                        chunk_rows(nrays, ngates, element_size(data)) as u64,
                        ngates as u64,
                    ],
                    shuffle: element_size(data) > 1,
                    deflate: options.deflate.map(|level| level.min(9)),
                }
            } else {
                NcStorage::Contiguous
            }
        }
        _ => NcStorage::Contiguous,
    };
    let mut attrs: Vec<(String, NcAttr)> = Vec::with_capacity(variable.attrs.len());
    for (name, value) in &variable.attrs {
        let name = sanitize_name(name);
        if RESERVED.contains(&name.as_str()) || attrs.iter().any(|(have, _)| *have == name) {
            continue;
        }
        if let Some(value) = attr(value) {
            attrs.push((name, value));
        }
    }
    Ok(NcVariable {
        name: sanitize_name(&variable.name),
        dims,
        data,
        attrs,
        storage,
    })
}
