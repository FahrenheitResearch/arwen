//! Variable-level read methods for classic NetCDF files.
//!
//! Provides convenience methods for reading variable data from a `ClassicFile`,
//! with type-checked access and support for both record and non-record variables.

use ndarray::ArrayD;

use crate::error::{Error, Result};
use crate::types::{
    checked_mul_u64, checked_shape_elements, checked_usize_from_u64, NcType, NcVariable,
};

use super::data::{self, compute_record_stride, NcReadType};
use super::ClassicFile;

#[derive(Clone, Debug)]
enum ResolvedClassicSelectionDim {
    Index(u64),
    Slice {
        start: u64,
        step: u64,
        count: usize,
        is_full_unit_stride: bool,
    },
}

impl ResolvedClassicSelectionDim {
    fn is_full_unit_stride(&self) -> bool {
        matches!(
            self,
            Self::Slice {
                is_full_unit_stride: true,
                ..
            }
        )
    }
}

#[derive(Clone, Debug)]
struct ResolvedClassicSelection {
    dims: Vec<ResolvedClassicSelectionDim>,
    result_shape: Vec<usize>,
    result_elements: usize,
}

struct BlockReadContext<'a> {
    file_data: &'a [u8],
    var_name: &'a str,
    base_offset: usize,
    plan: &'a ContiguousSelectionPlan,
}

struct ContiguousSelectionPlan {
    dims: Vec<ResolvedClassicSelectionDim>,
    strides: Vec<u64>,
    tail_start: usize,
    block_elements: usize,
    block_bytes: usize,
    elem_size: u64,
}

struct RecordSliceContext<'a> {
    file_data: &'a [u8],
    var_name: &'a str,
    base_offset: usize,
    record_stride: usize,
    inner_resolved: &'a ResolvedClassicSelection,
    inner_plan: &'a ContiguousSelectionPlan,
}

impl ClassicFile {
    /// Read a variable's data as an ndarray of the specified type.
    ///
    /// The type parameter `T` must match the variable's NetCDF type. For example,
    /// use `f32` for NC_FLOAT variables and `f64` for NC_DOUBLE variables.
    pub fn read_variable<T: NcReadType>(&self, name: &str) -> Result<ArrayD<T>> {
        let var = self.find_variable(name)?;

        // Check type compatibility.
        let expected = T::nc_type();
        if var.dtype != expected {
            return Err(Error::TypeMismatch {
                expected: format!("{:?}", expected),
                actual: format!("{:?}", var.dtype),
            });
        }

        let file_data = self.data.as_slice();

        if var.is_record_var {
            let record_stride = compute_record_stride(&self.root_group.variables);
            data::read_record_variable(file_data, var, self.numrecs, record_stride)
        } else {
            data::read_non_record_variable(file_data, var)
        }
    }

    /// Read a variable's data with automatic type promotion to f64.
    ///
    /// This reads any numeric variable and converts all values to f64,
    /// which is convenient for analysis but may lose precision for i64/u64.
    pub fn read_variable_as_f64(&self, name: &str) -> Result<ArrayD<f64>> {
        let var = self.find_variable(name)?;
        let file_data = self.data.as_slice();

        match var.dtype {
            NcType::Byte => {
                let arr = self.read_typed_variable::<i8>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Short => {
                let arr = self.read_typed_variable::<i16>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Int => {
                let arr = self.read_typed_variable::<i32>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Float => {
                let arr = self.read_typed_variable::<f32>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Double => self.read_typed_variable::<f64>(var, file_data),
            NcType::UByte => {
                let arr = self.read_typed_variable::<u8>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UShort => {
                let arr = self.read_typed_variable::<u16>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UInt => {
                let arr = self.read_typed_variable::<u32>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Int64 => {
                let arr = self.read_typed_variable::<i64>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UInt64 => {
                let arr = self.read_typed_variable::<u64>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Char => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: "Char".to_string(),
            }),
            NcType::String => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: "String".to_string(),
            }),
            _ => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: format!("{:?}", var.dtype),
            }),
        }
    }

    /// Read a char variable as a String (or Vec<String> for multi-dimensional).
    pub fn read_variable_as_string(&self, name: &str) -> Result<String> {
        let mut strings = self.read_variable_as_strings(name)?;
        match strings.len() {
            1 => Ok(strings.swap_remove(0)),
            0 => Err(Error::InvalidData(format!(
                "variable '{}' contains no string elements",
                name
            ))),
            count => Err(Error::InvalidData(format!(
                "variable '{}' contains {count} strings; use read_variable_as_strings()",
                name
            ))),
        }
    }

    /// Read a char variable as a flat vector of strings.
    ///
    /// For 2-D and higher char arrays, the last dimension is interpreted as the
    /// string length and the leading dimensions are flattened.
    pub fn read_variable_as_strings(&self, name: &str) -> Result<Vec<String>> {
        let var = self.find_variable(name)?;
        if var.dtype != NcType::Char {
            return Err(Error::TypeMismatch {
                expected: "Char".to_string(),
                actual: format!("{:?}", var.dtype),
            });
        }

        let file_data = self.data.as_slice();
        let arr = self.read_typed_variable::<u8>(var, file_data)?;
        let bytes: Vec<u8> = arr.iter().copied().collect();
        decode_char_variable_strings(var, &bytes)
    }

    /// Read a slice (hyperslab) of a variable.
    ///
    /// Classic variables are read directly from the on-disk byte ranges for
    /// arbitrary selections.
    pub fn read_variable_slice<T: NcReadType>(
        &self,
        name: &str,
        selection: &crate::types::NcSliceInfo,
    ) -> Result<ArrayD<T>> {
        let var = self.find_variable(name)?;
        let expected = T::nc_type();
        if var.dtype != expected {
            return Err(Error::TypeMismatch {
                expected: format!("{:?}", expected),
                actual: format!("{:?}", var.dtype),
            });
        }
        let file_data = self.data.as_slice();
        let resolved = resolve_classic_selection(
            var,
            selection,
            if var.is_record_var { self.numrecs } else { 0 },
        )?;

        if !var.is_record_var {
            return read_non_record_variable_slice_direct(file_data, var, &resolved);
        }

        let record_stride = compute_record_stride(&self.root_group.variables);
        read_record_variable_slice_direct(file_data, var, self.numrecs, record_stride, &resolved)
    }

    /// Read a slice with automatic type promotion to f64.
    pub fn read_variable_slice_as_f64(
        &self,
        name: &str,
        selection: &crate::types::NcSliceInfo,
    ) -> Result<ArrayD<f64>> {
        let var = self.find_variable(name)?;

        macro_rules! slice_promoted {
            ($ty:ty) => {{
                let sliced = self.read_variable_slice::<$ty>(name, selection)?;
                Ok(sliced.mapv(|v| v as f64))
            }};
        }

        match var.dtype {
            NcType::Byte => slice_promoted!(i8),
            NcType::Short => slice_promoted!(i16),
            NcType::Int => slice_promoted!(i32),
            NcType::Float => slice_promoted!(f32),
            NcType::Double => slice_promoted!(f64),
            NcType::UByte => slice_promoted!(u8),
            NcType::UShort => slice_promoted!(u16),
            NcType::UInt => slice_promoted!(u32),
            NcType::Int64 => slice_promoted!(i64),
            NcType::UInt64 => slice_promoted!(u64),
            NcType::Char => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: "Char".to_string(),
            }),
            _ => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: format!("{:?}", var.dtype),
            }),
        }
    }

    /// Internal: find a variable by name.
    fn find_variable(&self, name: &str) -> Result<&NcVariable> {
        self.root_group
            .variables
            .iter()
            .find(|v| v.name == name)
            .ok_or_else(|| Error::VariableNotFound(name.to_string()))
    }

    /// Internal: read a variable with the correct record handling.
    fn read_typed_variable<T: NcReadType>(
        &self,
        var: &NcVariable,
        file_data: &[u8],
    ) -> Result<ArrayD<T>> {
        if var.is_record_var {
            let record_stride = compute_record_stride(&self.root_group.variables);
            data::read_record_variable(file_data, var, self.numrecs, record_stride)
        } else {
            data::read_non_record_variable(file_data, var)
        }
    }
}

fn decode_char_variable_strings(var: &NcVariable, bytes: &[u8]) -> Result<Vec<String>> {
    let shape = var.shape();
    if shape.len() <= 1 {
        return Ok(vec![decode_char_string(bytes)]);
    }

    let string_len = checked_usize_from_u64(
        *shape
            .last()
            .ok_or_else(|| Error::InvalidData("char variable missing string axis".into()))?,
        "char string length",
    )?;
    let string_count_u64 = checked_shape_elements(&shape[..shape.len() - 1], "char string count")?;
    let string_count = checked_usize_from_u64(string_count_u64, "char string count")?;
    let expected_bytes = string_count.checked_mul(string_len).ok_or_else(|| {
        Error::InvalidData("char string byte count exceeds platform usize".to_string())
    })?;

    if bytes.len() < expected_bytes {
        return Err(Error::InvalidData(format!(
            "char variable '{}' data too short: need {} bytes, have {}",
            var.name,
            expected_bytes,
            bytes.len()
        )));
    }

    if string_len == 0 {
        return Ok(vec![String::new(); string_count]);
    }

    Ok(bytes[..expected_bytes]
        .chunks_exact(string_len)
        .map(decode_char_string)
        .collect())
}

fn decode_char_string(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim_end_matches('\0')
        .to_string()
}

fn variable_shape_for_selection(var: &NcVariable, numrecs: u64) -> Vec<u64> {
    let mut shape = var.shape();
    if var.is_record_var && !shape.is_empty() {
        shape[0] = numrecs;
    }
    shape
}

fn row_major_strides(shape: &[u64], context: &str) -> Result<Vec<u64>> {
    let ndim = shape.len();
    if ndim == 0 {
        return Ok(Vec::new());
    }

    let mut strides = vec![1u64; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = checked_mul_u64(strides[i + 1], shape[i + 1], context)?;
    }
    Ok(strides)
}

fn resolve_classic_selection(
    var: &NcVariable,
    selection: &crate::types::NcSliceInfo,
    numrecs: u64,
) -> Result<ResolvedClassicSelection> {
    use crate::types::NcSliceInfoElem;

    let shape = variable_shape_for_selection(var, numrecs);
    if selection.selections.len() != shape.len() {
        return Err(Error::InvalidData(format!(
            "selection has {} dimensions but variable '{}' has {}",
            selection.selections.len(),
            var.name,
            shape.len()
        )));
    }

    let mut dims = Vec::with_capacity(shape.len());
    let mut result_shape = Vec::new();
    let mut result_elements = 1usize;

    for (dim, (sel, &dim_size)) in selection.selections.iter().zip(shape.iter()).enumerate() {
        match sel {
            NcSliceInfoElem::Index(idx) => {
                if *idx >= dim_size {
                    return Err(Error::InvalidData(format!(
                        "index {} out of bounds for dimension {} (size {})",
                        idx, dim, dim_size
                    )));
                }
                dims.push(ResolvedClassicSelectionDim::Index(*idx));
            }
            NcSliceInfoElem::Slice { start, end, step } => {
                if *step == 0 {
                    return Err(Error::InvalidData("slice step cannot be 0".to_string()));
                }
                if *start > dim_size {
                    return Err(Error::InvalidData(format!(
                        "slice start {} out of bounds for dimension {} (size {})",
                        start, dim, dim_size
                    )));
                }

                let actual_end = if *end == u64::MAX {
                    dim_size
                } else {
                    (*end).min(dim_size)
                };
                let count_u64 = if *start >= actual_end {
                    0
                } else {
                    (actual_end - *start).div_ceil(*step)
                };
                let count = checked_usize_from_u64(count_u64, "classic slice result dimension")?;

                result_shape.push(count);
                result_elements = result_elements.checked_mul(count).ok_or_else(|| {
                    Error::InvalidData(
                        "classic slice result element count exceeds platform usize".to_string(),
                    )
                })?;
                dims.push(ResolvedClassicSelectionDim::Slice {
                    start: *start,
                    step: *step,
                    count,
                    is_full_unit_stride: *start == 0 && actual_end == dim_size && *step == 1,
                });
            }
        }
    }

    Ok(ResolvedClassicSelection {
        dims,
        result_shape,
        result_elements,
    })
}

fn read_non_record_variable_slice_direct<T: NcReadType>(
    file_data: &[u8],
    var: &NcVariable,
    resolved: &ResolvedClassicSelection,
) -> Result<ArrayD<T>> {
    let shape = variable_shape_for_selection(var, 0);
    let base_offset = checked_usize_from_u64(var.data_offset, "classic slice data offset")?;
    build_array_from_contiguous_selection::<T>(file_data, &var.name, base_offset, &shape, resolved)
}

fn read_record_variable_slice_direct<T: NcReadType>(
    file_data: &[u8],
    var: &NcVariable,
    numrecs: u64,
    record_stride: u64,
    resolved: &ResolvedClassicSelection,
) -> Result<ArrayD<T>> {
    use ndarray::IxDyn;

    if resolved.result_elements == 0 {
        return ArrayD::from_shape_vec(IxDyn(&resolved.result_shape), Vec::new())
            .map_err(|e| Error::InvalidData(format!("failed to create array: {e}")));
    }

    let shape = variable_shape_for_selection(var, numrecs);
    let inner_shape = &shape[1..];
    let inner_dims = resolved.dims[1..].to_vec();
    let inner_resolved = ResolvedClassicSelection {
        result_shape: selection_result_shape(&inner_dims),
        result_elements: selection_result_elements(&inner_dims)?,
        dims: inner_dims,
    };
    let inner_plan = build_contiguous_selection_plan::<T>(inner_shape, &inner_resolved.dims)?;
    let base_offset = checked_usize_from_u64(var.data_offset, "classic slice data offset")?;
    let record_stride = checked_usize_from_u64(record_stride, "classic record stride")?;
    let mut values = Vec::with_capacity(resolved.result_elements);
    let context = RecordSliceContext {
        file_data,
        var_name: &var.name,
        base_offset,
        record_stride,
        inner_resolved: &inner_resolved,
        inner_plan: &inner_plan,
    };

    match &resolved.dims[0] {
        ResolvedClassicSelectionDim::Index(record) => {
            append_one_record_slice::<T>(&context, *record, &mut values)?
        }
        ResolvedClassicSelectionDim::Slice {
            start, step, count, ..
        } => {
            for ordinal in 0..*count {
                let record = start
                    .checked_add(checked_mul_u64(
                        ordinal as u64,
                        *step,
                        "classic record slice coordinate",
                    )?)
                    .ok_or_else(|| {
                        Error::InvalidData(
                            "classic record slice coordinate exceeds u64".to_string(),
                        )
                    })?;
                append_one_record_slice::<T>(&context, record, &mut values)?;
            }
        }
    }

    debug_assert_eq!(values.len(), resolved.result_elements);
    ArrayD::from_shape_vec(IxDyn(&resolved.result_shape), values)
        .map_err(|e| Error::InvalidData(format!("failed to create array: {e}")))
}

fn selection_result_shape(dims: &[ResolvedClassicSelectionDim]) -> Vec<usize> {
    dims.iter()
        .filter_map(|dim| match dim {
            ResolvedClassicSelectionDim::Index(_) => None,
            ResolvedClassicSelectionDim::Slice { count, .. } => Some(*count),
        })
        .collect()
}

fn selection_result_elements(dims: &[ResolvedClassicSelectionDim]) -> Result<usize> {
    let mut elements = 1usize;
    for dim in dims {
        if let ResolvedClassicSelectionDim::Slice { count, .. } = dim {
            elements = elements.checked_mul(*count).ok_or_else(|| {
                Error::InvalidData(
                    "classic slice result element count exceeds platform usize".to_string(),
                )
            })?;
        }
    }
    Ok(elements)
}

fn build_array_from_contiguous_selection<T: NcReadType>(
    file_data: &[u8],
    var_name: &str,
    base_offset: usize,
    shape: &[u64],
    resolved: &ResolvedClassicSelection,
) -> Result<ArrayD<T>> {
    use ndarray::IxDyn;

    let plan = build_contiguous_selection_plan::<T>(shape, &resolved.dims)?;
    let values = read_contiguous_selection_values_with_plan::<T>(
        file_data,
        var_name,
        base_offset,
        &plan,
        resolved.result_elements,
    )?;
    ArrayD::from_shape_vec(IxDyn(&resolved.result_shape), values)
        .map_err(|e| Error::InvalidData(format!("failed to create array: {e}")))
}

fn build_contiguous_selection_plan<T: NcReadType>(
    shape: &[u64],
    dims: &[ResolvedClassicSelectionDim],
) -> Result<ContiguousSelectionPlan> {
    let strides = row_major_strides(shape, "classic slice stride")?;
    let tail_start = dims
        .iter()
        .rposition(|dim| !dim.is_full_unit_stride())
        .map_or(0, |idx| idx + 1);
    let block_elements_u64 = checked_shape_elements(
        &shape[tail_start..],
        "classic slice contiguous block element count",
    )?;
    let block_elements = checked_usize_from_u64(
        block_elements_u64,
        "classic slice contiguous block element count",
    )?;
    let block_bytes = checked_usize_from_u64(
        checked_mul_u64(
            block_elements_u64,
            T::element_size() as u64,
            "classic slice contiguous block size in bytes",
        )?,
        "classic slice contiguous block size in bytes",
    )?;

    Ok(ContiguousSelectionPlan {
        dims: dims.to_vec(),
        strides,
        tail_start,
        block_elements,
        block_bytes,
        elem_size: T::element_size() as u64,
    })
}

fn read_contiguous_selection_values_with_plan<T: NcReadType>(
    file_data: &[u8],
    var_name: &str,
    base_offset: usize,
    plan: &ContiguousSelectionPlan,
    result_elements: usize,
) -> Result<Vec<T>> {
    if result_elements == 0 {
        return Ok(Vec::new());
    }

    let mut values = Vec::with_capacity(result_elements);
    let context = BlockReadContext {
        file_data,
        var_name,
        base_offset,
        plan,
    };
    read_selected_blocks_recursive::<T>(0, 0, &context, &mut values)?;
    Ok(values)
}

fn append_one_record_slice<T: NcReadType>(
    context: &RecordSliceContext<'_>,
    record: u64,
    values: &mut Vec<T>,
) -> Result<()> {
    let record = checked_usize_from_u64(record, "classic record index")?;
    let record_offset = context
        .base_offset
        .checked_add(record.checked_mul(context.record_stride).ok_or_else(|| {
            Error::InvalidData("classic record byte offset exceeds platform usize".to_string())
        })?)
        .ok_or_else(|| {
            Error::InvalidData("classic record byte offset exceeds platform usize".to_string())
        })?;
    let mut decoded = read_contiguous_selection_values_with_plan::<T>(
        context.file_data,
        context.var_name,
        record_offset,
        context.inner_plan,
        context.inner_resolved.result_elements,
    )?;
    values.append(&mut decoded);
    Ok(())
}

fn read_selected_blocks_recursive<T: NcReadType>(
    level: usize,
    current_offset: u64,
    context: &BlockReadContext<'_>,
    values: &mut Vec<T>,
) -> Result<()> {
    if level == context.plan.tail_start {
        let byte_offset = checked_usize_from_u64(
            checked_mul_u64(
                current_offset,
                context.plan.elem_size,
                "classic slice element byte offset",
            )?,
            "classic slice element byte offset",
        )?;
        let start = context
            .base_offset
            .checked_add(byte_offset)
            .ok_or_else(|| {
                Error::InvalidData("classic slice byte offset exceeds platform usize".to_string())
            })?;
        let end = start.checked_add(context.plan.block_bytes).ok_or_else(|| {
            Error::InvalidData("classic slice byte range exceeds platform usize".to_string())
        })?;
        if end > context.file_data.len() {
            return Err(Error::InvalidData(format!(
                "variable '{}' slice data extends beyond file",
                context.var_name
            )));
        }

        let mut decoded =
            T::decode_bulk_be(&context.file_data[start..end], context.plan.block_elements)?;
        values.append(&mut decoded);
        return Ok(());
    }

    match &context.plan.dims[level] {
        ResolvedClassicSelectionDim::Index(idx) => read_selected_blocks_recursive::<T>(
            level + 1,
            current_offset
                .checked_add(checked_mul_u64(
                    *idx,
                    context.plan.strides[level],
                    "classic slice logical element offset",
                )?)
                .ok_or_else(|| {
                    Error::InvalidData(
                        "classic slice logical element offset exceeds u64".to_string(),
                    )
                })?,
            context,
            values,
        ),
        ResolvedClassicSelectionDim::Slice {
            start, step, count, ..
        } => {
            let start = *start;
            let step = *step;
            let count = *count;
            for ordinal in 0..count {
                let coord = start
                    .checked_add(checked_mul_u64(
                        ordinal as u64,
                        step,
                        "classic slice coordinate",
                    )?)
                    .ok_or_else(|| {
                        Error::InvalidData("classic slice coordinate exceeds u64".to_string())
                    })?;
                read_selected_blocks_recursive::<T>(
                    level + 1,
                    current_offset
                        .checked_add(checked_mul_u64(
                            coord,
                            context.plan.strides[level],
                            "classic slice logical element offset",
                        )?)
                        .ok_or_else(|| {
                            Error::InvalidData(
                                "classic slice logical element offset exceeds u64".to_string(),
                            )
                        })?,
                    context,
                    values,
                )?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NcDimension;

    fn char_variable(shape: &[u64]) -> NcVariable {
        NcVariable {
            name: "chars".to_string(),
            dimensions: shape
                .iter()
                .enumerate()
                .map(|(i, &size)| NcDimension {
                    name: format!("d{i}"),
                    size,
                    is_unlimited: false,
                })
                .collect(),
            dtype: NcType::Char,
            attributes: vec![],
            data_offset: 0,
            _data_size: 0,
            is_record_var: false,
            record_size: 0,
        }
    }

    #[test]
    fn test_decode_char_variable_strings_1d() {
        let var = char_variable(&[5]);
        let strings = decode_char_variable_strings(&var, b"alpha").unwrap();
        assert_eq!(strings, vec!["alpha"]);
    }

    #[test]
    fn test_decode_char_variable_strings_2d() {
        let var = char_variable(&[2, 5]);
        let strings = decode_char_variable_strings(&var, b"alphabeta\0").unwrap();
        assert_eq!(strings, vec!["alpha", "beta"]);
    }
}
