use std::error::Error;
use std::io;
use std::path::Path;

use hdf5_reader::{Datatype, SliceInfo as H5SliceInfo, SliceInfoElem as H5SliceInfoElem};

#[derive(Debug, Clone, PartialEq)]
pub struct ScaledVariable {
    pub name: String,
    pub shape: Vec<usize>,
    pub units: Option<String>,
    pub values: Vec<f32>,
}

pub fn open_goes_netcdf_lossy(path: impl AsRef<Path>) -> Result<netcrust::File, Box<dyn Error>> {
    let options = netcrust::NcOpenOptions {
        metadata_mode: netcrust::NcMetadataMode::Lossy,
        ..Default::default()
    };
    Ok(netcrust::File::open_with_options(path, options)?)
}

pub fn read_scaled_f32(
    file: &netcrust::File,
    name: &str,
) -> Result<ScaledVariable, Box<dyn Error>> {
    let Some(variable) = file.variable(name) else {
        let path = file
            .path()
            .ok_or_else(|| boxed_error(format!("variable not found: {name}")))?;
        return read_scaled_f32_hdf5(path, name);
    };
    let scale = variable
        .attribute("scale_factor")
        .and_then(|attr| attr.as_f64())
        .unwrap_or(1.0);
    let offset = variable
        .attribute("add_offset")
        .and_then(|attr| attr.as_f64())
        .unwrap_or(0.0);
    let fill = variable
        .attribute("_FillValue")
        .and_then(|attr| attr.as_f64());
    let valid_range = variable
        .attribute("valid_range")
        .and_then(|attr| attr.value().as_f64_vec())
        .and_then(|values| match values.as_slice() {
            [min, max, ..] => Some((*min, *max)),
            _ => None,
        });
    let units = variable
        .attribute("units")
        .and_then(|attr| attr.as_string())
        .map(str::to_string);

    let array = variable.array_f64()?;
    let shape = array.shape().to_vec();
    let values = scale_values(array.into_values(), scale, offset, fill, valid_range);

    Ok(ScaledVariable {
        name: name.to_string(),
        shape,
        units,
        values,
    })
}

pub fn read_scaled_f32_window(
    file: &netcrust::File,
    name: &str,
    y_start: usize,
    y_count: usize,
    x_start: usize,
    x_count: usize,
) -> Result<ScaledVariable, Box<dyn Error>> {
    if y_count == 0 || x_count == 0 {
        return Err(boxed_error(format!(
            "empty NetCDF window requested for {name}: y_count={y_count} x_count={x_count}"
        )));
    }
    let Some(variable) = file.variable(name) else {
        let path = file
            .path()
            .ok_or_else(|| boxed_error(format!("variable not found: {name}")))?;
        return read_scaled_f32_hdf5_window(path, name, y_start, y_count, x_start, x_count);
    };
    if variable.ndim() != 2 {
        return Err(boxed_error(format!(
            "window reads require a 2D variable; {name} has shape {:?}",
            variable.shape()
        )));
    }
    let scale = variable
        .attribute("scale_factor")
        .and_then(|attr| attr.as_f64())
        .unwrap_or(1.0);
    let offset = variable
        .attribute("add_offset")
        .and_then(|attr| attr.as_f64())
        .unwrap_or(0.0);
    let fill = variable
        .attribute("_FillValue")
        .and_then(|attr| attr.as_f64());
    let valid_range = variable
        .attribute("valid_range")
        .and_then(|attr| attr.value().as_f64_vec())
        .and_then(|values| match values.as_slice() {
            [min, max, ..] => Some((*min, *max)),
            _ => None,
        });
    let units = variable
        .attribute("units")
        .and_then(|attr| attr.as_string())
        .map(str::to_string);

    let selection = netcrust::NcSliceInfo {
        selections: vec![
            netcrust::NcSliceInfoElem::Slice {
                start: y_start as u64,
                end: y_start.saturating_add(y_count) as u64,
                step: 1,
            },
            netcrust::NcSliceInfoElem::Slice {
                start: x_start as u64,
                end: x_start.saturating_add(x_count) as u64,
                step: 1,
            },
        ],
    };
    let array = match variable.array_f64_slice(&selection) {
        Ok(array) => array,
        Err(err) => {
            let path = file.path().ok_or_else(|| {
                boxed_error(format!(
                    "failed to read {name} slice and file path is unavailable"
                ))
            })?;
            return read_scaled_f32_hdf5_window(
                path, name, y_start, y_count, x_start, x_count,
            )
            .map_err(|fallback_err| {
                boxed_error(format!(
                    "failed to read {name} slice through netcrust ({err}); HDF5 fallback failed: {fallback_err}"
                ))
            });
        }
    };
    let shape = array.shape().to_vec();
    let values = scale_values(array.into_values(), scale, offset, fill, valid_range);

    Ok(ScaledVariable {
        name: name.to_string(),
        shape,
        units,
        values,
    })
}

fn read_scaled_f32_hdf5(path: &Path, name: &str) -> Result<ScaledVariable, Box<dyn Error>> {
    let file = hdf5_reader::Hdf5File::open(path)?;
    let dataset = file.dataset(name)?;
    let scale = hdf5_attr_f64(&dataset, "scale_factor").unwrap_or(1.0);
    let offset = hdf5_attr_f64(&dataset, "add_offset").unwrap_or(0.0);
    let fill = hdf5_attr_f64(&dataset, "_FillValue");
    let valid_range =
        hdf5_attr_f64_vec(&dataset, "valid_range").and_then(|values| match values.as_slice() {
            [min, max, ..] => Some((*min, *max)),
            _ => None,
        });
    let units = dataset
        .attribute("units")
        .ok()
        .and_then(|attr| attr.read_string().ok());
    let shape = dataset
        .shape()
        .iter()
        .map(|&value| usize::try_from(value))
        .collect::<Result<Vec<_>, _>>()?;
    let values = scale_values(
        hdf5_dataset_values_f64(&dataset)?,
        scale,
        offset,
        fill,
        valid_range,
    );

    Ok(ScaledVariable {
        name: name.to_string(),
        shape,
        units,
        values,
    })
}

fn read_scaled_f32_hdf5_window(
    path: &Path,
    name: &str,
    y_start: usize,
    y_count: usize,
    x_start: usize,
    x_count: usize,
) -> Result<ScaledVariable, Box<dyn Error>> {
    let file = hdf5_reader::Hdf5File::open(path)?;
    let dataset = file.dataset(name)?;
    if dataset.ndim() != 2 {
        return Err(boxed_error(format!(
            "window reads require a 2D HDF5 dataset; {name} has shape {:?}",
            dataset.shape()
        )));
    }
    let scale = hdf5_attr_f64(&dataset, "scale_factor").unwrap_or(1.0);
    let offset = hdf5_attr_f64(&dataset, "add_offset").unwrap_or(0.0);
    let fill = hdf5_attr_f64(&dataset, "_FillValue");
    let valid_range =
        hdf5_attr_f64_vec(&dataset, "valid_range").and_then(|values| match values.as_slice() {
            [min, max, ..] => Some((*min, *max)),
            _ => None,
        });
    let units = dataset
        .attribute("units")
        .ok()
        .and_then(|attr| attr.read_string().ok());
    let selection = H5SliceInfo {
        selections: vec![
            H5SliceInfoElem::Slice {
                start: y_start as u64,
                end: y_start.saturating_add(y_count) as u64,
                step: 1,
            },
            H5SliceInfoElem::Slice {
                start: x_start as u64,
                end: x_start.saturating_add(x_count) as u64,
                step: 1,
            },
        ],
    };
    let array = hdf5_dataset_values_f64_slice(&dataset, &selection)?;
    let shape = vec![y_count, x_count];
    let values = scale_values(array, scale, offset, fill, valid_range);

    Ok(ScaledVariable {
        name: name.to_string(),
        shape,
        units,
        values,
    })
}

fn scale_values(
    values: Vec<f64>,
    scale: f64,
    offset: f64,
    fill: Option<f64>,
    valid_range: Option<(f64, f64)>,
) -> Vec<f32> {
    values
        .into_iter()
        .map(|value| {
            if !value.is_finite()
                || fill.is_some_and(|fill| (value - fill).abs() < 0.5)
                || valid_range.is_some_and(|(min, max)| value < min || value > max)
            {
                f32::NAN
            } else {
                (value * scale + offset) as f32
            }
        })
        .collect()
}

fn hdf5_dataset_values_f64(dataset: &hdf5_reader::Dataset) -> Result<Vec<f64>, Box<dyn Error>> {
    match dataset.dtype() {
        Datatype::FloatingPoint { size: 4, .. } => Ok(dataset
            .read_array::<f32>()?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FloatingPoint { size: 8, .. } => {
            Ok(dataset.read_array::<f64>()?.iter().copied().collect())
        }
        Datatype::FixedPoint {
            size: 1,
            signed: true,
            ..
        } => Ok(dataset
            .read_array::<i8>()?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 1,
            signed: false,
            ..
        } => Ok(dataset
            .read_array::<u8>()?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 2,
            signed: true,
            ..
        } => Ok(dataset
            .read_array::<i16>()?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 2,
            signed: false,
            ..
        } => Ok(dataset
            .read_array::<u16>()?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 4,
            signed: true,
            ..
        } => Ok(dataset
            .read_array::<i32>()?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 4,
            signed: false,
            ..
        } => Ok(dataset
            .read_array::<u32>()?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 8,
            signed: true,
            ..
        } => Ok(dataset
            .read_array::<i64>()?
            .iter()
            .map(|&value| value as f64)
            .collect()),
        Datatype::FixedPoint {
            size: 8,
            signed: false,
            ..
        } => Ok(dataset
            .read_array::<u64>()?
            .iter()
            .map(|&value| value as f64)
            .collect()),
        dtype => Err(boxed_error(format!(
            "unsupported HDF5 numeric dataset type for {}: {dtype:?}",
            dataset.name()
        ))),
    }
}

fn hdf5_dataset_values_f64_slice(
    dataset: &hdf5_reader::Dataset,
    selection: &H5SliceInfo,
) -> Result<Vec<f64>, Box<dyn Error>> {
    match dataset.dtype() {
        Datatype::FloatingPoint { size: 4, .. } => Ok(dataset
            .read_slice::<f32>(selection)?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FloatingPoint { size: 8, .. } => Ok(dataset
            .read_slice::<f64>(selection)?
            .iter()
            .copied()
            .collect()),
        Datatype::FixedPoint {
            size: 1,
            signed: true,
            ..
        } => Ok(dataset
            .read_slice::<i8>(selection)?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 1,
            signed: false,
            ..
        } => Ok(dataset
            .read_slice::<u8>(selection)?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 2,
            signed: true,
            ..
        } => Ok(dataset
            .read_slice::<i16>(selection)?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 2,
            signed: false,
            ..
        } => Ok(dataset
            .read_slice::<u16>(selection)?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 4,
            signed: true,
            ..
        } => Ok(dataset
            .read_slice::<i32>(selection)?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 4,
            signed: false,
            ..
        } => Ok(dataset
            .read_slice::<u32>(selection)?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 8,
            signed: true,
            ..
        } => Ok(dataset
            .read_slice::<i64>(selection)?
            .iter()
            .map(|&value| value as f64)
            .collect()),
        Datatype::FixedPoint {
            size: 8,
            signed: false,
            ..
        } => Ok(dataset
            .read_slice::<u64>(selection)?
            .iter()
            .map(|&value| value as f64)
            .collect()),
        dtype => Err(boxed_error(format!(
            "unsupported HDF5 numeric dataset type for {}: {dtype:?}",
            dataset.name()
        ))),
    }
}

fn hdf5_attr_f64(dataset: &hdf5_reader::Dataset, name: &str) -> Option<f64> {
    dataset
        .attribute(name)
        .ok()
        .and_then(|attr| attr.read_as_f64().ok())
}

fn hdf5_attr_f64_vec(dataset: &hdf5_reader::Dataset, name: &str) -> Option<Vec<f64>> {
    let attr = dataset.attribute(name).ok()?;
    match &attr.datatype {
        Datatype::FloatingPoint { size: 4, .. } => Some(
            attr.read_1d::<f32>()
                .ok()?
                .into_iter()
                .map(f64::from)
                .collect(),
        ),
        Datatype::FloatingPoint { size: 8, .. } => attr.read_1d::<f64>().ok(),
        Datatype::FixedPoint {
            size: 1,
            signed: true,
            ..
        } => Some(
            attr.read_1d::<i8>()
                .ok()?
                .into_iter()
                .map(f64::from)
                .collect(),
        ),
        Datatype::FixedPoint {
            size: 1,
            signed: false,
            ..
        } => Some(
            attr.read_1d::<u8>()
                .ok()?
                .into_iter()
                .map(f64::from)
                .collect(),
        ),
        Datatype::FixedPoint {
            size: 2,
            signed: true,
            ..
        } => Some(
            attr.read_1d::<i16>()
                .ok()?
                .into_iter()
                .map(f64::from)
                .collect(),
        ),
        Datatype::FixedPoint {
            size: 2,
            signed: false,
            ..
        } => Some(
            attr.read_1d::<u16>()
                .ok()?
                .into_iter()
                .map(f64::from)
                .collect(),
        ),
        Datatype::FixedPoint {
            size: 4,
            signed: true,
            ..
        } => Some(
            attr.read_1d::<i32>()
                .ok()?
                .into_iter()
                .map(f64::from)
                .collect(),
        ),
        Datatype::FixedPoint {
            size: 4,
            signed: false,
            ..
        } => Some(
            attr.read_1d::<u32>()
                .ok()?
                .into_iter()
                .map(f64::from)
                .collect(),
        ),
        Datatype::FixedPoint {
            size: 8,
            signed: true,
            ..
        } => Some(
            attr.read_1d::<i64>()
                .ok()?
                .into_iter()
                .map(|value| value as f64)
                .collect(),
        ),
        Datatype::FixedPoint {
            size: 8,
            signed: false,
            ..
        } => Some(
            attr.read_1d::<u64>()
                .ok()?
                .into_iter()
                .map(|value| value as f64)
                .collect(),
        ),
        _ => None,
    }
}

fn boxed_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}
