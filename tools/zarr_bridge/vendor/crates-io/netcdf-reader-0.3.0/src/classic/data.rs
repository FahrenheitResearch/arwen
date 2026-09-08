//! Data reading for classic (CDF-1/2/5) NetCDF files.
//!
//! Two layout types:
//! - **Non-record variables**: contiguous data at the offset stored in the variable header.
//! - **Record variables**: data is interleaved across records. Each record contains one
//!   slice from every record variable, in the order they appear in the header. The total
//!   record size is the sum of all record variables' vsize values (each padded to 4-byte
//!   boundary in CDF-1/2).

use ndarray::{ArrayD, IxDyn};

use crate::error::{Error, Result};
use crate::types::{NcType, NcVariable};

/// Trait for types that can be read from classic NetCDF data.
pub trait NcReadType: Clone + Default + Send + 'static {
    /// The NetCDF type this Rust type corresponds to.
    fn nc_type() -> NcType;

    /// Read a single element from big-endian bytes.
    fn from_be_bytes(bytes: &[u8]) -> Result<Self>;

    /// Size in bytes of one element.
    fn element_size() -> usize;

    /// Bulk decode `count` elements from a contiguous big-endian byte slice.
    ///
    /// Default implementation falls back to per-element decoding. Types with
    /// multi-byte elements override this with an optimized bulk path using
    /// `chunks_exact` + byte-swap (on LE hosts) or `copy_nonoverlapping`
    /// (on BE hosts).
    fn decode_bulk_be(raw: &[u8], count: usize) -> Result<Vec<Self>> {
        let elem_size = Self::element_size();
        let needed = count * elem_size;
        if raw.len() < needed {
            return Err(Error::InvalidData(format!(
                "need {} bytes for {} elements, got {}",
                needed,
                count,
                raw.len()
            )));
        }
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * elem_size;
            values.push(Self::from_be_bytes(&raw[start..start + elem_size])?);
        }
        Ok(values)
    }
}

macro_rules! impl_nc_read_type {
    ($ty:ty, $nc_type:expr, $size:expr) => {
        impl NcReadType for $ty {
            fn nc_type() -> NcType {
                $nc_type
            }

            fn from_be_bytes(bytes: &[u8]) -> Result<Self> {
                if bytes.len() < $size {
                    return Err(Error::InvalidData(format!(
                        "need {} bytes for {}, got {}",
                        $size,
                        stringify!($ty),
                        bytes.len()
                    )));
                }
                let mut arr = [0u8; $size];
                arr.copy_from_slice(&bytes[..$size]);
                Ok(<$ty>::from_be_bytes(arr))
            }

            fn element_size() -> usize {
                $size
            }

            fn decode_bulk_be(raw: &[u8], count: usize) -> Result<Vec<Self>> {
                let total_bytes = count * $size;
                if raw.len() < total_bytes {
                    return Err(Error::InvalidData(format!(
                        "need {} bytes for {} elements of {}, got {}",
                        total_bytes,
                        count,
                        stringify!($ty),
                        raw.len()
                    )));
                }
                let bytes = &raw[..total_bytes];
                #[cfg(target_endian = "big")]
                {
                    // Native BE: memcpy is safe for any element size.
                    let mut values = Vec::<$ty>::with_capacity(count);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            values.as_mut_ptr() as *mut u8,
                            total_bytes,
                        );
                        values.set_len(count);
                    }
                    Ok(values)
                }
                #[cfg(target_endian = "little")]
                {
                    // LE host reading BE data: chunks_exact + byte-swap.
                    Ok(bytes
                        .chunks_exact($size)
                        .map(|chunk| {
                            let mut arr = [0u8; $size];
                            arr.copy_from_slice(chunk);
                            <$ty>::from_be_bytes(arr)
                        })
                        .collect())
                }
            }
        }
    };
}

impl_nc_read_type!(i8, NcType::Byte, 1);
impl_nc_read_type!(i16, NcType::Short, 2);
impl_nc_read_type!(i32, NcType::Int, 4);
impl_nc_read_type!(f32, NcType::Float, 4);
impl_nc_read_type!(f64, NcType::Double, 8);
impl_nc_read_type!(u8, NcType::UByte, 1);
impl_nc_read_type!(u16, NcType::UShort, 2);
impl_nc_read_type!(u32, NcType::UInt, 4);
impl_nc_read_type!(i64, NcType::Int64, 8);
impl_nc_read_type!(u64, NcType::UInt64, 8);

/// Read the entire data for a non-record variable into an ndarray.
///
/// The data is located at a contiguous region starting at `var.data_offset`
/// with total size `var.data_size`.
pub fn read_non_record_variable<T: NcReadType>(
    file_data: &[u8],
    var: &NcVariable,
) -> Result<ArrayD<T>> {
    if var.is_record_var {
        return Err(Error::InvalidData(
            "use read_record_variable for record variables".to_string(),
        ));
    }

    let offset = crate::types::checked_usize_from_u64(var.data_offset, "variable data offset")?;
    let total_elements = crate::types::checked_usize_from_u64(
        var.checked_num_elements()?,
        "variable element count",
    )?;
    let elem_size = T::element_size();
    let total_bytes = total_elements.checked_mul(elem_size).ok_or_else(|| {
        Error::InvalidData(format!(
            "variable '{}' size in bytes exceeds platform usize",
            var.name
        ))
    })?;

    let end = offset.checked_add(total_bytes).ok_or_else(|| {
        Error::InvalidData(format!(
            "variable '{}' byte range exceeds platform usize",
            var.name
        ))
    })?;
    if end > file_data.len() {
        return Err(Error::InvalidData(format!(
            "variable '{}' data extends beyond file: offset={}, size={}, file_len={}",
            var.name,
            offset,
            total_bytes,
            file_data.len()
        )));
    }

    let data_slice = &file_data[offset..end];
    let values = T::decode_bulk_be(data_slice, total_elements)?;

    let shape: Vec<usize> = var
        .shape()
        .iter()
        .map(|&s| crate::types::checked_usize_from_u64(s, "variable dimension"))
        .collect::<Result<Vec<_>>>()?;
    if shape.is_empty() {
        // Scalar variable.
        ArrayD::from_shape_vec(IxDyn(&[]), values)
    } else {
        ArrayD::from_shape_vec(IxDyn(&shape), values)
    }
    .map_err(|e| Error::InvalidData(format!("failed to create array: {}", e)))
}

/// Read the entire data for a record variable into an ndarray.
///
/// Record variables are interleaved: for each of `numrecs` records, every record
/// variable contributes `record_size` bytes (padded to 4-byte alignment for CDF-1/2).
/// The `record_stride` is the total size of one record across all record variables.
///
/// Parameters:
/// - `file_data`: the raw file bytes
/// - `var`: the record variable to read
/// - `numrecs`: number of records (from the file header)
/// - `record_stride`: total bytes per record (sum of all record variables' padded vsizes)
pub fn read_record_variable<T: NcReadType>(
    file_data: &[u8],
    var: &NcVariable,
    numrecs: u64,
    record_stride: u64,
) -> Result<ArrayD<T>> {
    if !var.is_record_var {
        return Err(Error::InvalidData(
            "use read_non_record_variable for non-record variables".to_string(),
        ));
    }

    let elem_size = T::element_size();
    let base_offset =
        crate::types::checked_usize_from_u64(var.data_offset, "record variable data offset")?;
    let numrecs_usize = crate::types::checked_usize_from_u64(numrecs, "record count")?;
    let record_stride_usize = crate::types::checked_usize_from_u64(record_stride, "record stride")?;

    // Shape: the first dimension is the unlimited dimension, replaced by numrecs.
    let mut shape: Vec<usize> = var
        .shape()
        .iter()
        .map(|&s| crate::types::checked_usize_from_u64(s, "record variable dimension"))
        .collect::<Result<Vec<_>>>()?;
    if shape.is_empty() {
        return Err(Error::InvalidData(
            "record variable must have at least one dimension".to_string(),
        ));
    }
    shape[0] = numrecs_usize;

    // Number of elements per record (product of all dims except the first).
    let elements_per_record: usize = shape[1..].iter().product::<usize>().max(1);
    let bytes_per_record = elements_per_record.checked_mul(elem_size).ok_or_else(|| {
        Error::InvalidData(format!(
            "record variable '{}' bytes per record exceed platform usize",
            var.name
        ))
    })?;
    let total_elements = numrecs_usize
        .checked_mul(elements_per_record)
        .ok_or_else(|| {
            Error::InvalidData(format!(
                "record variable '{}' element count exceeds platform usize",
                var.name
            ))
        })?;

    let mut values = Vec::with_capacity(total_elements);

    for rec in 0..numrecs_usize {
        let rec_offset = base_offset
            .checked_add(rec.checked_mul(record_stride_usize).ok_or_else(|| {
                Error::InvalidData(format!(
                    "record variable '{}' byte offset exceeds platform usize",
                    var.name
                ))
            })?)
            .ok_or_else(|| {
                Error::InvalidData(format!(
                    "record variable '{}' byte offset exceeds platform usize",
                    var.name
                ))
            })?;
        let rec_end = rec_offset.checked_add(bytes_per_record).ok_or_else(|| {
            Error::InvalidData(format!(
                "record variable '{}' record range exceeds platform usize",
                var.name
            ))
        })?;
        if rec_end > file_data.len() {
            return Err(Error::InvalidData(format!(
                "record {} for variable '{}' extends beyond file",
                rec, var.name
            )));
        }
        let rec_slice = &file_data[rec_offset..rec_end];
        let rec_values = T::decode_bulk_be(rec_slice, elements_per_record)?;
        values.extend(rec_values);
    }

    ArrayD::from_shape_vec(IxDyn(&shape), values)
        .map_err(|e| Error::InvalidData(format!("failed to create array: {}", e)))
}

/// Compute the record stride: total bytes per record across all record variables.
///
/// Each record variable's per-record contribution is its `record_size` (already stored
/// as vsize from the header), padded to 4-byte boundary.
pub fn compute_record_stride(variables: &[NcVariable]) -> u64 {
    variables
        .iter()
        .filter(|v| v.is_record_var)
        .map(|v| {
            let size = v.record_size;
            // Pad each variable's per-record size to 4-byte boundary.
            let rem = size % 4;
            if rem == 0 {
                size
            } else {
                size + (4 - rem)
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NcDimension;

    #[test]
    fn test_read_non_record_1d_float() {
        // Create a fake file with 3 floats starting at offset 100.
        let mut file_data = vec![0u8; 200];
        let values = [1.0f32, 2.0f32, 3.0f32];
        for (i, &v) in values.iter().enumerate() {
            let bytes = v.to_be_bytes();
            file_data[100 + i * 4..100 + i * 4 + 4].copy_from_slice(&bytes);
        }

        let var = NcVariable {
            name: "temp".to_string(),
            dimensions: vec![NcDimension {
                name: "x".to_string(),
                size: 3,
                is_unlimited: false,
            }],
            dtype: NcType::Float,
            attributes: vec![],
            data_offset: 100,
            _data_size: 12,
            is_record_var: false,
            record_size: 0,
        };

        let arr: ArrayD<f32> = read_non_record_variable(&file_data, &var).unwrap();
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr[[0]], 1.0f32);
        assert_eq!(arr[[1]], 2.0f32);
        assert_eq!(arr[[2]], 3.0f32);
    }

    #[test]
    fn test_read_non_record_2d_int() {
        // 2x3 array of i32 at offset 0
        let values: Vec<i32> = vec![10, 20, 30, 40, 50, 60];
        let mut file_data = Vec::new();
        for &v in &values {
            file_data.extend_from_slice(&v.to_be_bytes());
        }

        let var = NcVariable {
            name: "grid".to_string(),
            dimensions: vec![
                NcDimension {
                    name: "y".to_string(),
                    size: 2,
                    is_unlimited: false,
                },
                NcDimension {
                    name: "x".to_string(),
                    size: 3,
                    is_unlimited: false,
                },
            ],
            dtype: NcType::Int,
            attributes: vec![],
            data_offset: 0,
            _data_size: 24,
            is_record_var: false,
            record_size: 0,
        };

        let arr: ArrayD<i32> = read_non_record_variable(&file_data, &var).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 10);
        assert_eq!(arr[[0, 2]], 30);
        assert_eq!(arr[[1, 0]], 40);
        assert_eq!(arr[[1, 2]], 60);
    }

    #[test]
    fn test_compute_record_stride() {
        let vars = vec![
            NcVariable {
                name: "a".to_string(),
                dimensions: vec![],
                dtype: NcType::Float,
                attributes: vec![],
                data_offset: 0,
                _data_size: 0,
                is_record_var: true,
                record_size: 20, // 5 floats
            },
            NcVariable {
                name: "b".to_string(),
                dimensions: vec![],
                dtype: NcType::Short,
                attributes: vec![],
                data_offset: 0,
                _data_size: 0,
                is_record_var: true,
                record_size: 6, // 3 shorts -> padded to 8
            },
            NcVariable {
                name: "c".to_string(),
                dimensions: vec![],
                dtype: NcType::Double,
                attributes: vec![],
                data_offset: 0,
                _data_size: 100,
                is_record_var: false, // not a record var, should be excluded
                record_size: 0,
            },
        ];
        // a: 20 (already 4-aligned), b: 6 -> 8 = total 28
        assert_eq!(compute_record_stride(&vars), 28);
    }

    #[test]
    fn test_read_record_variable() {
        // Single record variable "temp" with shape [time, x] where x=2.
        // 3 records, each with 2 floats = 8 bytes per record.
        // Record stride = 8 (only one record var, already 4-aligned).
        let mut file_data = vec![0u8; 200];
        let base = 100usize;
        let record_values: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        for (rec, vals) in record_values.iter().enumerate() {
            for (i, &v) in vals.iter().enumerate() {
                let offset = base + rec * 8 + i * 4;
                file_data[offset..offset + 4].copy_from_slice(&v.to_be_bytes());
            }
        }

        let var = NcVariable {
            name: "temp".to_string(),
            dimensions: vec![
                NcDimension {
                    name: "time".to_string(),
                    size: 0, // unlimited
                    is_unlimited: true,
                },
                NcDimension {
                    name: "x".to_string(),
                    size: 2,
                    is_unlimited: false,
                },
            ],
            dtype: NcType::Float,
            attributes: vec![],
            data_offset: 100,
            _data_size: 0,
            is_record_var: true,
            record_size: 8,
        };

        let arr: ArrayD<f32> = read_record_variable(&file_data, &var, 3, 8).unwrap();
        assert_eq!(arr.shape(), &[3, 2]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[0, 1]], 2.0);
        assert_eq!(arr[[1, 0]], 3.0);
        assert_eq!(arr[[2, 1]], 6.0);
    }

    #[test]
    fn test_read_non_record_variable_rejects_element_count_overflow() {
        let var = NcVariable {
            name: "huge".to_string(),
            dimensions: vec![
                NcDimension {
                    name: "y".to_string(),
                    size: u64::MAX,
                    is_unlimited: false,
                },
                NcDimension {
                    name: "x".to_string(),
                    size: 2,
                    is_unlimited: false,
                },
            ],
            dtype: NcType::Float,
            attributes: vec![],
            data_offset: 0,
            _data_size: 0,
            is_record_var: false,
            record_size: 0,
        };

        let err = read_non_record_variable::<f32>(&[], &var).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn test_read_record_variable_rejects_record_offset_overflow() {
        let var = NcVariable {
            name: "huge_record".to_string(),
            dimensions: vec![
                NcDimension {
                    name: "time".to_string(),
                    size: 0,
                    is_unlimited: true,
                },
                NcDimension {
                    name: "x".to_string(),
                    size: 1,
                    is_unlimited: false,
                },
            ],
            dtype: NcType::Float,
            attributes: vec![],
            data_offset: u64::MAX,
            _data_size: 0,
            is_record_var: true,
            record_size: 4,
        };

        let err = read_record_variable::<f32>(&[], &var, 1, 4).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }
}
