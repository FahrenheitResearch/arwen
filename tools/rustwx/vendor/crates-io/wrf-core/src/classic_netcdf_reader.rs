//! Pure-Rust classic NetCDF/CDF reader adapter.
//!
//! WRF can write NetCDF classic 64-bit offset files (`CDF\x02`). This adapter
//! gives `WrfFile` the same small API as the pure HDF5 reader without using the
//! C NetCDF library.

use std::path::Path;

use netcdf_reader::{NcAttrValue, NcFile, NcSliceInfo, NcSliceInfoElem, NcType};

use crate::error::{WrfError, WrfResult};

pub struct ClassicNetcdfFile {
    nc: NcFile,
}

impl ClassicNetcdfFile {
    pub fn open<P: AsRef<Path>>(path: P) -> WrfResult<Self> {
        let nc = NcFile::open(path).map_err(nc_err)?;
        Ok(Self { nc })
    }

    pub fn has_dataset(&self, name: &str) -> bool {
        self.nc.variable(name).is_ok()
    }

    pub fn dataset_shape(&self, name: &str) -> WrfResult<Vec<usize>> {
        let var = self.nc.variable(name).map_err(nc_err)?;
        var.shape()
            .into_iter()
            .map(|value| usize::try_from(value).map_err(|_| shape_err(name, value)))
            .collect()
    }

    pub fn read_f64_slice(&self, name: &str, time_index: usize) -> WrfResult<Vec<f64>> {
        let var = self.nc.variable(name).map_err(nc_err)?;
        let shape = var.shape();
        let has_time_dim = var
            .dimensions()
            .first()
            .is_some_and(|dim| dim.is_unlimited || dim.name == "Time");

        let data = if has_time_dim && shape.len() >= 2 {
            if time_index as u64 >= shape[0] {
                return Err(WrfError::NetCdf(format!(
                    "time_index {time_index} out of range for variable `{name}` shape {shape:?}"
                )));
            }
            let mut selections = Vec::with_capacity(shape.len());
            selections.push(NcSliceInfoElem::Index(time_index as u64));
            selections.extend(shape.iter().skip(1).map(|&end| NcSliceInfoElem::Slice {
                start: 0,
                end,
                step: 1,
            }));
            self.nc
                .read_variable_slice_as_f64(name, &NcSliceInfo { selections })
                .map_err(nc_err)?
        } else {
            self.nc.read_variable_as_f64(name).map_err(nc_err)?
        };

        Ok(data.iter().copied().collect())
    }

    pub fn read_u8(&self, name: &str) -> WrfResult<Vec<u8>> {
        let var = self.nc.variable(name).map_err(nc_err)?;
        if *var.dtype() == NcType::Char {
            return self.read_char_as_padded_bytes(name, var.shape());
        }

        let data = self.nc.read_variable::<u8>(name).map_err(nc_err)?;
        Ok(data.iter().copied().collect())
    }

    pub fn global_attr_f64(&self, name: &str) -> WrfResult<f64> {
        self.nc
            .global_attribute(name)
            .map_err(nc_err)?
            .value
            .as_f64()
            .ok_or_else(|| WrfError::AttrType(format!("'{name}' is not numeric")))
    }

    pub fn global_attr_i32(&self, name: &str) -> WrfResult<i32> {
        let value = self.global_attr_f64(name)?;
        Ok(value as i32)
    }

    pub fn global_attr_string(&self, name: &str) -> WrfResult<String> {
        let attr = self.nc.global_attribute(name).map_err(nc_err)?;
        match &attr.value {
            NcAttrValue::Chars(value) => Ok(value.clone()),
            NcAttrValue::Strings(values) if values.len() == 1 => Ok(values[0].clone()),
            _ => attr
                .value
                .as_f64()
                .map(|value| value.to_string())
                .ok_or_else(|| WrfError::AttrType(format!("'{name}' cannot be read as string"))),
        }
    }

    fn read_char_as_padded_bytes(&self, name: &str, shape: Vec<u64>) -> WrfResult<Vec<u8>> {
        let strings = self.nc.read_variable_as_strings(name).map_err(nc_err)?;
        let string_len = shape
            .last()
            .copied()
            .ok_or_else(|| WrfError::DimMismatch(format!("char variable `{name}` has no shape")))?;
        let string_len = usize::try_from(string_len).map_err(|_| shape_err(name, string_len))?;
        let mut bytes = Vec::with_capacity(strings.len() * string_len);
        for value in strings {
            let raw = value.as_bytes();
            let copy_len = raw.len().min(string_len);
            bytes.extend_from_slice(&raw[..copy_len]);
            bytes.resize(bytes.len() + string_len - copy_len, 0);
        }
        Ok(bytes)
    }
}

fn nc_err(error: netcdf_reader::Error) -> WrfError {
    match error {
        netcdf_reader::Error::VariableNotFound(name) => WrfError::VarNotFound(name),
        netcdf_reader::Error::AttributeNotFound(name) => WrfError::AttrNotFound(name),
        netcdf_reader::Error::DimensionNotFound(name) => {
            WrfError::DimMismatch(format!("dimension '{name}' not found"))
        }
        other => WrfError::NetCdf(other.to_string()),
    }
}

fn shape_err(name: &str, value: u64) -> WrfError {
    WrfError::DimMismatch(format!(
        "dimension value {value} for variable `{name}` exceeds platform usize"
    ))
}
