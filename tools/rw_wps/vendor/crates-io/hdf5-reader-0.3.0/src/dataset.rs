use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::sync::{Arc, OnceLock};

use lru::LruCache;
use ndarray::{ArrayD, IxDyn};
use parking_lot::Mutex;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::attribute_api::{
    collect_attribute_messages_storage, decode_string, decode_varlen_byte_string,
    read_one_vlen_string_storage, resolve_vlen_bytes_storage, Attribute,
};
use crate::cache::{ChunkCache, ChunkKey};
use crate::chunk_index;
use crate::datatype_api::{dtype_element_size, H5Type};
use crate::error::{Error, Result};
use crate::filters::{self, FilterRegistry};
use crate::io::Cursor;
use crate::messages::attribute::AttributeMessage;
use crate::messages::dataspace::{DataspaceMessage, DataspaceType};
use crate::messages::datatype::{Datatype, StringSize};
use crate::messages::fill_value::{FillTime, FillValueMessage};
use crate::messages::filter_pipeline::FilterPipelineMessage;
use crate::messages::layout::{ChunkIndexing, DataLayout};
use crate::messages::HdfMessage;
use crate::object_header::ObjectHeader;
use crate::FileContext;

const HOT_FULL_DATASET_CACHE_MAX_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy)]
struct FlatBufferPtr {
    ptr: *mut u8,
    len: usize,
}

#[derive(Clone, Copy)]
struct ChunkCopyLayout<'a> {
    chunk_offsets: &'a [u64],
    chunk_shape: &'a [u64],
    dataset_shape: &'a [u64],
    dataset_strides: &'a [usize],
    chunk_strides: &'a [usize],
    elem_size: usize,
}

#[derive(Clone, Copy)]
struct UnitStrideCopyLayout<'a> {
    chunk_offsets: &'a [u64],
    chunk_shape: &'a [u64],
    dataset_shape: &'a [u64],
    resolved: &'a ResolvedSelection,
    chunk_strides: &'a [usize],
    result_strides: &'a [usize],
    elem_size: usize,
}

pub(crate) struct DatasetParseContext {
    pub(crate) context: Arc<FileContext>,
}

#[derive(Clone, Copy)]
struct ChunkEntrySelection<'a> {
    shape: &'a [u64],
    ndim: usize,
    elem_size: usize,
    chunk_bounds: Option<(&'a [u64], &'a [u64])>,
}

unsafe impl Send for FlatBufferPtr {}

unsafe impl Sync for FlatBufferPtr {}

impl FlatBufferPtr {
    #[cfg(feature = "rayon")]
    #[inline(always)]
    unsafe fn copy_chunk(self, chunk_data: &[u8], layout: ChunkCopyLayout<'_>) {
        copy_chunk_to_flat_with_strides_ptr(chunk_data, self, layout);
    }

    #[cfg(feature = "rayon")]
    #[inline(always)]
    unsafe fn copy_selected(
        self,
        chunk_data: &[u8],
        dim_indices: &[Vec<(usize, usize)>],
        chunk_strides: &[usize],
        result_strides: &[usize],
        elem_size: usize,
        ndim: usize,
    ) {
        copy_selected_elements_ptr(
            chunk_data,
            self.ptr,
            self.len,
            dim_indices,
            chunk_strides,
            result_strides,
            elem_size,
            ndim,
        );
    }

    #[cfg(feature = "rayon")]
    #[inline(always)]
    unsafe fn copy_unit_stride_chunk_overlap(
        self,
        chunk_data: &[u8],
        layout: UnitStrideCopyLayout<'_>,
    ) -> Result<()> {
        copy_unit_stride_chunk_overlap_ptr(chunk_data, self, layout)
    }
}

/// Hyperslab selection for reading slices of datasets.
#[derive(Debug, Clone)]
pub struct SliceInfo {
    pub selections: Vec<SliceInfoElem>,
}

/// A single dimension's selection.
#[derive(Debug, Clone)]
pub enum SliceInfoElem {
    /// Select a single index (reduces dimensionality).
    Index(u64),
    /// Select a range with optional step.
    Slice { start: u64, end: u64, step: u64 },
}

#[derive(Clone, Debug)]
struct ResolvedSelectionDim {
    start: u64,
    end: u64,
    step: u64,
    count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ChunkEntryCacheKey {
    index_address: u64,
    first_chunk: SmallVec<[u64; 4]>,
    last_chunk: SmallVec<[u64; 4]>,
}

impl ResolvedSelectionDim {
    fn chunk_index_range(&self, chunk_extent: u64) -> Option<(u64, u64)> {
        if self.count == 0 {
            return None;
        }

        Some((self.start / chunk_extent, (self.end - 1) / chunk_extent))
    }
}

#[derive(Clone, Debug)]
struct ResolvedSelection {
    dims: Vec<ResolvedSelectionDim>,
    result_shape: Vec<usize>,
    result_elements: usize,
}

impl ResolvedSelection {
    fn result_dims_with_collapsed(&self) -> Vec<usize> {
        self.dims.iter().map(|dim| dim.count).collect()
    }

    fn is_unit_stride(&self) -> bool {
        self.dims.iter().all(|dim| dim.step == 1)
    }
}

impl SliceInfo {
    /// Create a selection that reads everything.
    pub fn all(ndim: usize) -> Self {
        SliceInfo {
            selections: vec![
                SliceInfoElem::Slice {
                    start: 0,
                    end: u64::MAX,
                    step: 1,
                };
                ndim
            ],
        }
    }
}

fn checked_usize(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        Error::InvalidData(format!(
            "{context} value {value} exceeds platform usize capacity"
        ))
    })
}

fn checked_mul_usize(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidData(format!("{context} exceeds platform usize capacity")))
}

fn checked_add_usize(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidData(format!("{context} exceeds platform usize capacity")))
}

fn checked_shape_elements_usize(shape: &[u64], context: &str) -> Result<usize> {
    let mut total = 1usize;
    for &dim in shape {
        total = checked_mul_usize(total, checked_usize(dim, context)?, context)?;
    }
    Ok(total)
}

fn expected_chunk_count(first_chunk: &[u64], last_chunk: &[u64]) -> Result<usize> {
    let mut total = 1usize;
    for (&first, &last) in first_chunk.iter().zip(last_chunk.iter()) {
        let dim_count = checked_usize(last - first + 1, "selected chunk count")?;
        total = checked_mul_usize(total, dim_count, "selected chunk count")?;
    }
    Ok(total)
}

fn full_dataset_chunk_count(shape: &[u64], chunk_shape: &[u64]) -> Result<usize> {
    let mut total = 1usize;
    for (&dim, &chunk) in shape.iter().zip(chunk_shape.iter()) {
        let chunk_count = checked_usize(dim.div_ceil(chunk), "full dataset chunk count")?;
        total = checked_mul_usize(total, chunk_count, "full dataset chunk count")?;
    }
    Ok(total)
}

fn row_major_strides(shape: &[u64], context: &str) -> Result<Vec<usize>> {
    let ndim = shape.len();
    if ndim == 0 {
        return Ok(Vec::new());
    }

    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        let next_extent = checked_usize(shape[i + 1], context)?;
        strides[i] = checked_mul_usize(strides[i + 1], next_extent, context)?;
    }
    Ok(strides)
}

fn assume_init_u8_vec(mut buffer: Vec<MaybeUninit<u8>>) -> Vec<u8> {
    let ptr = buffer.as_mut_ptr() as *mut u8;
    let len = buffer.len();
    let capacity = buffer.capacity();
    std::mem::forget(buffer);
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

fn assume_init_vec<T>(mut buffer: Vec<MaybeUninit<T>>) -> Vec<T> {
    let ptr = buffer.as_mut_ptr() as *mut T;
    let len = buffer.len();
    let capacity = buffer.capacity();
    std::mem::forget(buffer);
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

fn normalize_selection(selection: &SliceInfo, shape: &[u64]) -> Result<ResolvedSelection> {
    if selection.selections.len() != shape.len() {
        return Err(Error::InvalidData(format!(
            "slice has {} dimensions but dataset has {}",
            selection.selections.len(),
            shape.len()
        )));
    }

    let mut dims = Vec::with_capacity(shape.len());
    let mut result_shape = Vec::new();
    let mut result_elements = 1usize;

    for (i, sel) in selection.selections.iter().enumerate() {
        let dim_size = shape[i];
        match sel {
            SliceInfoElem::Index(idx) => {
                if *idx >= dim_size {
                    return Err(Error::SliceOutOfBounds {
                        dim: i,
                        index: *idx,
                        size: dim_size,
                    });
                }
                dims.push(ResolvedSelectionDim {
                    start: *idx,
                    end: *idx + 1,
                    step: 1,
                    count: 1,
                });
            }
            SliceInfoElem::Slice { start, end, step } => {
                if *step == 0 {
                    return Err(Error::InvalidData("slice step cannot be 0".into()));
                }
                if *start > dim_size {
                    return Err(Error::SliceOutOfBounds {
                        dim: i,
                        index: *start,
                        size: dim_size,
                    });
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
                let count = checked_usize(count_u64, "slice element count")?;

                dims.push(ResolvedSelectionDim {
                    start: *start,
                    end: actual_end,
                    step: *step,
                    count,
                });
                result_shape.push(count);
                result_elements =
                    checked_mul_usize(result_elements, count, "slice result element count")?;
            }
        }
    }

    Ok(ResolvedSelection {
        dims,
        result_shape,
        result_elements,
    })
}

/// A dataset within an HDF5 file.
pub struct Dataset {
    context: Arc<FileContext>,
    pub(crate) name: String,
    pub(crate) data_address: u64,
    pub(crate) dataspace: DataspaceMessage,
    pub(crate) datatype: Datatype,
    pub(crate) layout: DataLayout,
    pub(crate) fill_value: Option<FillValueMessage>,
    pub(crate) filters: Option<FilterPipelineMessage>,
    pub(crate) attributes: Vec<AttributeMessage>,
    pub(crate) chunk_cache: Arc<ChunkCache>,
    chunk_entry_cache: Arc<Mutex<LruCache<ChunkEntryCacheKey, Arc<Vec<chunk_index::ChunkEntry>>>>>,
    full_chunk_entries: Arc<OnceLock<Arc<Vec<chunk_index::ChunkEntry>>>>,
    full_dataset_bytes: Arc<OnceLock<Arc<Vec<u8>>>>,
    pub(crate) filter_registry: Arc<FilterRegistry>,
}

pub(crate) struct DatasetTemplate {
    name: String,
    data_address: u64,
    dataspace: DataspaceMessage,
    datatype: Datatype,
    layout: DataLayout,
    fill_value: Option<FillValueMessage>,
    filters: Option<FilterPipelineMessage>,
    attributes: Vec<AttributeMessage>,
    chunk_entry_cache: Arc<Mutex<LruCache<ChunkEntryCacheKey, Arc<Vec<chunk_index::ChunkEntry>>>>>,
    full_chunk_entries: Arc<OnceLock<Arc<Vec<chunk_index::ChunkEntry>>>>,
    full_dataset_bytes: Arc<OnceLock<Arc<Vec<u8>>>>,
}

impl Dataset {
    pub(crate) fn from_template(context: Arc<FileContext>, template: Arc<DatasetTemplate>) -> Self {
        Dataset {
            chunk_cache: context.chunk_cache.clone(),
            filter_registry: context.filter_registry.clone(),
            context,
            name: template.name.clone(),
            data_address: template.data_address,
            dataspace: template.dataspace.clone(),
            datatype: template.datatype.clone(),
            layout: template.layout.clone(),
            fill_value: template.fill_value.clone(),
            filters: template.filters.clone(),
            attributes: template.attributes.clone(),
            chunk_entry_cache: template.chunk_entry_cache.clone(),
            full_chunk_entries: template.full_chunk_entries.clone(),
            full_dataset_bytes: template.full_dataset_bytes.clone(),
        }
    }

    pub(crate) fn template(&self) -> Arc<DatasetTemplate> {
        Arc::new(DatasetTemplate {
            name: self.name.clone(),
            data_address: self.data_address,
            dataspace: self.dataspace.clone(),
            datatype: self.datatype.clone(),
            layout: self.layout.clone(),
            fill_value: self.fill_value.clone(),
            filters: self.filters.clone(),
            attributes: self.attributes.clone(),
            chunk_entry_cache: self.chunk_entry_cache.clone(),
            full_chunk_entries: self.full_chunk_entries.clone(),
            full_dataset_bytes: self.full_dataset_bytes.clone(),
        })
    }

    pub(crate) fn from_parsed_header(
        context: DatasetParseContext,
        address: u64,
        name: String,
        header: &ObjectHeader,
    ) -> Result<Self> {
        let mut dataspace: Option<DataspaceMessage> = None;
        let mut datatype: Option<Datatype> = None;
        let mut layout: Option<DataLayout> = None;
        let mut fill_value: Option<FillValueMessage> = None;
        let mut filter_pipeline: Option<FilterPipelineMessage> = None;
        let attributes = collect_attribute_messages_storage(
            header,
            context.context.storage.as_ref(),
            context.context.superblock.offset_size,
            context.context.superblock.length_size,
        )?;

        for msg in &header.messages {
            match msg {
                HdfMessage::Dataspace(ds) => dataspace = Some(ds.clone()),
                HdfMessage::Datatype(dt) => datatype = Some(dt.datatype.clone()),
                HdfMessage::DataLayout(dl) => layout = Some(dl.layout.clone()),
                HdfMessage::FillValue(fv) => fill_value = Some(fv.clone()),
                HdfMessage::FilterPipeline(fp) => filter_pipeline = Some(fp.clone()),
                _ => {}
            }
        }

        let dataspace =
            dataspace.ok_or_else(|| Error::InvalidData("dataset missing dataspace".into()))?;
        let dt = datatype.ok_or_else(|| Error::InvalidData("dataset missing datatype".into()))?;
        let layout =
            layout.ok_or_else(|| Error::InvalidData("dataset missing data layout".into()))?;
        let layout = normalize_layout(layout, &dataspace);
        let attr_fill_value = attributes
            .iter()
            .find(|attr| attr.name == "_FillValue" && attr.dataspace.num_elements() == 1)
            .map(|attr| FillValueMessage {
                defined: !attr.raw_data.is_empty(),
                fill_time: FillTime::IfSet,
                value: Some(attr.raw_data.clone()),
            });
        let fill_value = match fill_value {
            Some(existing) if existing.value.is_some() => Some(existing),
            _ => attr_fill_value,
        };

        Ok(Dataset {
            context: context.context.clone(),
            name,
            data_address: address,
            dataspace,
            datatype: dt,
            layout,
            fill_value,
            filters: filter_pipeline,
            attributes,
            chunk_cache: context.context.chunk_cache.clone(),
            chunk_entry_cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(32).unwrap()))),
            full_chunk_entries: Arc::new(OnceLock::new()),
            full_dataset_bytes: Arc::new(OnceLock::new()),
            filter_registry: context.context.filter_registry.clone(),
        })
    }

    /// Dataset name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The object header address used to parse this dataset.
    /// Useful as an opaque identifier or for NC4 data_offset.
    pub fn address(&self) -> u64 {
        self.data_address
    }

    /// Shape of the dataset (dimensions).
    pub fn shape(&self) -> &[u64] {
        &self.dataspace.dims
    }

    /// Datatype of the dataset.
    pub fn dtype(&self) -> &Datatype {
        &self.datatype
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dataspace.dims.len()
    }

    fn offset_size(&self) -> u8 {
        self.context.superblock.offset_size
    }

    fn length_size(&self) -> u8 {
        self.context.superblock.length_size
    }

    /// Maximum dimension sizes, if defined. `u64::MAX` indicates unlimited.
    pub fn max_dims(&self) -> Option<&[u64]> {
        self.dataspace.max_dims.as_deref()
    }

    /// Chunk dimensions, if the dataset is chunked.
    pub fn chunks(&self) -> Option<Vec<u32>> {
        match &self.layout {
            DataLayout::Chunked { dims, .. } => Some(dims.clone()),
            _ => None,
        }
    }

    /// Fill value, if defined.
    pub fn fill_value(&self) -> Option<&FillValueMessage> {
        self.fill_value.as_ref()
    }

    /// Dataset attributes.
    pub fn attributes(&self) -> Vec<Attribute> {
        self.attributes
            .iter()
            .map(|a| attribute_from_message_storage(a, self.context.as_ref()))
            .collect()
    }

    /// Find an attribute by name.
    pub fn attribute(&self, name: &str) -> Result<Attribute> {
        self.attributes
            .iter()
            .find(|a| a.name == name)
            .map(|a| attribute_from_message_storage(a, self.context.as_ref()))
            .ok_or_else(|| Error::AttributeNotFound(name.to_string()))
    }

    /// Read a scalar string dataset or single string element.
    ///
    /// Use [`Dataset::read_strings`] when the dataset contains multiple strings.
    pub fn read_string(&self) -> Result<String> {
        let mut strings = self.read_strings()?;
        match strings.len() {
            1 => Ok(strings.swap_remove(0)),
            0 => Err(Error::InvalidData(format!(
                "dataset '{}' contains no string elements",
                self.name
            ))),
            count => Err(Error::InvalidData(format!(
                "dataset '{}' contains {count} string elements; use read_strings()",
                self.name
            ))),
        }
    }

    /// Read all string elements from a string-typed dataset.
    pub fn read_strings(&self) -> Result<Vec<String>> {
        match &self.datatype {
            Datatype::String {
                size: StringSize::Fixed(len),
                encoding,
                padding,
            } => {
                let raw = self.read_raw_bytes()?;
                let elem_size = *len as usize;
                let count = checked_usize(self.num_elements(), "dataset string element count")?;
                let expected_bytes =
                    checked_mul_usize(count, elem_size, "dataset string byte size")?;
                if raw.len() < expected_bytes {
                    return Err(Error::InvalidData(format!(
                        "dataset '{}' string data too short: need {} bytes, have {}",
                        self.name,
                        expected_bytes,
                        raw.len()
                    )));
                }

                let mut strings = Vec::with_capacity(count);
                for i in 0..count {
                    let start = i * elem_size;
                    let end = start + elem_size;
                    strings.push(decode_string(&raw[start..end], *padding, *encoding)?);
                }
                Ok(strings)
            }
            Datatype::String {
                size: StringSize::Variable,
                encoding,
                padding,
            } => {
                let raw = self.read_raw_bytes()?;
                let count = checked_usize(self.num_elements(), "dataset string element count")?;
                let ref_size = 4 + self.offset_size() as usize + 4;
                let expected_bytes =
                    checked_mul_usize(count, ref_size, "dataset string reference byte size")?;
                if raw.len() < expected_bytes {
                    return Err(Error::InvalidData(format!(
                        "dataset '{}' vlen string data too short: need {} bytes, have {}",
                        self.name,
                        expected_bytes,
                        raw.len()
                    )));
                }

                let mut strings = Vec::with_capacity(count);
                for i in 0..count {
                    let offset = i * ref_size;
                    strings.push(read_one_vlen_string_storage(
                        &raw,
                        offset,
                        self.context.storage.as_ref(),
                        self.offset_size(),
                        self.length_size(),
                        *padding,
                        *encoding,
                    )?);
                }
                Ok(strings)
            }
            Datatype::VarLen { base } => {
                if !matches!(base.as_ref(), Datatype::FixedPoint { size: 1, .. }) {
                    return Err(Error::TypeMismatch {
                        expected: "String dataset".into(),
                        actual: format!("{:?}", self.datatype),
                    });
                }

                let raw = self.read_raw_bytes()?;
                let count = checked_usize(self.num_elements(), "dataset string element count")?;
                let ref_size = 4 + self.offset_size() as usize + 4;
                let expected_bytes =
                    checked_mul_usize(count, ref_size, "dataset string reference byte size")?;
                if raw.len() < expected_bytes {
                    return Err(Error::InvalidData(format!(
                        "dataset '{}' vlen byte string data too short: need {} bytes, have {}",
                        self.name,
                        expected_bytes,
                        raw.len()
                    )));
                }

                let mut strings = Vec::with_capacity(count);
                for i in 0..count {
                    let offset = i * ref_size;
                    let ref_bytes = &raw[offset..offset + ref_size];
                    let value = resolve_vlen_bytes_storage(
                        ref_bytes,
                        self.context.storage.as_ref(),
                        self.offset_size(),
                        self.length_size(),
                    )
                    .unwrap_or_default();
                    strings.push(decode_varlen_byte_string(&value)?);
                }
                Ok(strings)
            }
            _ => Err(Error::TypeMismatch {
                expected: "String dataset".into(),
                actual: format!("{:?}", self.datatype),
            }),
        }
    }

    /// Total number of elements in the dataset.
    pub fn num_elements(&self) -> u64 {
        if self.dataspace.dims.is_empty() {
            match self.dataspace.dataspace_type {
                DataspaceType::Scalar => 1,
                DataspaceType::Null => 0,
                DataspaceType::Simple => 0,
            }
        } else {
            self.dataspace.dims.iter().product()
        }
    }

    /// Read the entire dataset into an n-dimensional array.
    pub fn read_array<T: H5Type>(&self) -> Result<ArrayD<T>> {
        let result = match &self.layout {
            DataLayout::Compact { data } => self.read_compact::<T>(data),
            DataLayout::Contiguous { address, size } => self.read_contiguous::<T>(*address, *size),
            DataLayout::Chunked {
                address,
                dims,
                element_size,
                chunk_indexing,
            } => self.read_chunked::<T>(*address, dims, *element_size, chunk_indexing.as_ref()),
        };
        result.map_err(|e| e.with_context(&self.name))
    }

    /// Read the entire dataset using internal chunk-level parallelism when possible.
    ///
    /// Non-chunked datasets fall back to `read_array`.
    #[cfg(feature = "rayon")]
    pub fn read_array_parallel<T: H5Type>(&self) -> Result<ArrayD<T>> {
        match &self.layout {
            DataLayout::Chunked {
                address,
                dims,
                element_size,
                chunk_indexing,
            } => self.read_chunked_parallel::<T>(
                *address,
                dims,
                *element_size,
                chunk_indexing.as_ref(),
            ),
            _ => self.read_array::<T>(),
        }
    }

    /// Read the entire dataset using the provided Rayon thread pool.
    ///
    /// Non-chunked datasets fall back to `read_array`.
    #[cfg(feature = "rayon")]
    pub fn read_array_in_pool<T: H5Type>(&self, pool: &rayon::ThreadPool) -> Result<ArrayD<T>> {
        match &self.layout {
            DataLayout::Chunked {
                address,
                dims,
                element_size,
                chunk_indexing,
            } => pool.install(|| {
                self.read_chunked_parallel::<T>(
                    *address,
                    dims,
                    *element_size,
                    chunk_indexing.as_ref(),
                )
            }),
            _ => self.read_array::<T>(),
        }
    }

    /// Read a hyperslab of the dataset using chunk-level parallelism when possible.
    ///
    /// Chunked datasets decompress overlapping chunks in parallel via Rayon.
    /// Non-chunked layouts fall back to `read_slice`.
    #[cfg(feature = "rayon")]
    pub fn read_slice_parallel<T: H5Type>(&self, selection: &SliceInfo) -> Result<ArrayD<T>> {
        let resolved = normalize_selection(selection, &self.dataspace.dims)?;

        match &self.layout {
            DataLayout::Chunked {
                address,
                dims,
                element_size,
                chunk_indexing,
            } => self.read_chunked_slice_parallel::<T>(
                *address,
                dims,
                *element_size,
                chunk_indexing.as_ref(),
                selection,
                &resolved,
            ),
            _ => self.read_slice::<T>(selection),
        }
    }

    /// Read a hyperslab of the dataset.
    pub fn read_slice<T: H5Type>(&self, selection: &SliceInfo) -> Result<ArrayD<T>> {
        let resolved = normalize_selection(selection, &self.dataspace.dims)?;

        match &self.layout {
            DataLayout::Contiguous { address, size } => {
                self.read_contiguous_slice::<T>(*address, *size, selection, &resolved)
            }
            DataLayout::Compact { data } => self.read_compact_slice::<T>(data, selection),
            DataLayout::Chunked {
                address,
                dims,
                element_size,
                chunk_indexing,
            } => self.read_chunked_slice::<T>(
                *address,
                dims,
                *element_size,
                chunk_indexing.as_ref(),
                selection,
                &resolved,
            ),
        }
    }

    fn read_compact<T: H5Type>(&self, data: &[u8]) -> Result<ArrayD<T>> {
        self.decode_raw_data::<T>(data)
    }

    fn read_raw_bytes(&self) -> Result<Vec<u8>> {
        let elem_size = dtype_element_size(&self.datatype);
        let total_elements = checked_usize(self.num_elements(), "dataset element count")?;
        let total_bytes = checked_mul_usize(total_elements, elem_size, "dataset size in bytes")?;

        let result = match &self.layout {
            DataLayout::Compact { data } => Ok(self.normalize_raw_bytes(data, total_bytes)),
            DataLayout::Contiguous { address, size } => {
                self.read_contiguous_bytes(*address, *size, total_bytes)
            }
            DataLayout::Chunked {
                address,
                dims,
                element_size: _,
                chunk_indexing,
            } => self.read_chunked_bytes(*address, dims, chunk_indexing.as_ref(), total_bytes),
        };

        result.map_err(|e| e.with_context(&self.name))
    }

    fn read_contiguous<T: H5Type>(&self, address: u64, size: u64) -> Result<ArrayD<T>> {
        if Cursor::is_undefined_offset(address, self.offset_size()) || size == 0 {
            // Dataset with no data written — return fill values
            return self.make_fill_array::<T>();
        }

        let sz = checked_usize(size, "contiguous dataset size")?;
        let raw = self.context.read_range(address, sz)?;
        self.decode_raw_data::<T>(raw.as_ref())
    }

    fn read_contiguous_bytes(
        &self,
        address: u64,
        size: u64,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        if Cursor::is_undefined_offset(address, self.offset_size()) || size == 0 {
            return Ok(self.make_output_buffer(total_bytes));
        }

        let sz = checked_usize(size, "contiguous dataset size")?;
        let raw = self.context.read_range(address, sz)?;
        Ok(self.normalize_raw_bytes(raw.as_ref(), total_bytes))
    }

    fn read_chunked<T: H5Type>(
        &self,
        index_address: u64,
        chunk_dims: &[u32],
        _element_size: u32,
        chunk_indexing: Option<&ChunkIndexing>,
    ) -> Result<ArrayD<T>> {
        if Cursor::is_undefined_offset(index_address, self.offset_size()) {
            return self.make_fill_array::<T>();
        }

        let ndim = self.ndim();
        let shape = &self.dataspace.dims;
        let elem_size = dtype_element_size(&self.datatype);
        let chunk_shape: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
        let dataset_strides = row_major_strides(shape, "dataset stride")?;
        let chunk_strides = row_major_strides(&chunk_shape, "chunk stride")?;

        // Allocate output initialized from the dataset's fill value.
        let total_elements = checked_usize(self.num_elements(), "dataset element count")?;
        let total_bytes = checked_mul_usize(total_elements, elem_size, "dataset size in bytes")?;

        let entries = self.collect_chunk_entries(
            index_address,
            chunk_dims,
            chunk_indexing,
            ChunkEntrySelection {
                shape,
                ndim,
                elem_size,
                chunk_bounds: None,
            },
        )?;

        let full_chunk_coverage = entries.len() == full_dataset_chunk_count(shape, &chunk_shape)?;
        if full_chunk_coverage {
            let hot_full_dataset_bytes = if total_bytes <= HOT_FULL_DATASET_CACHE_MAX_BYTES {
                self.full_dataset_bytes.get().cloned()
            } else {
                None
            };
            if let Some(cached_bytes) = hot_full_dataset_bytes {
                return self.decode_raw_data::<T>(&cached_bytes);
            }
            if T::native_copy_compatible(&self.datatype) && std::mem::size_of::<T>() == elem_size {
                let mut result_values: Vec<MaybeUninit<T>> =
                    std::iter::repeat_with(MaybeUninit::<T>::uninit)
                        .take(total_elements)
                        .collect();
                let result_ptr = result_values.as_mut_ptr() as *mut u8;
                let result_len = checked_mul_usize(
                    result_values.len(),
                    std::mem::size_of::<T>(),
                    "typed dataset size in bytes",
                )?;

                for entry in &entries {
                    let chunk_data =
                        self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)?;
                    unsafe {
                        copy_chunk_to_flat_with_strides_ptr(
                            &chunk_data,
                            FlatBufferPtr {
                                ptr: result_ptr,
                                len: result_len,
                            },
                            ChunkCopyLayout {
                                chunk_offsets: &entry.offsets,
                                chunk_shape: &chunk_shape,
                                dataset_shape: shape,
                                dataset_strides: &dataset_strides,
                                chunk_strides: &chunk_strides,
                                elem_size,
                            },
                        );
                    }
                }

                if total_bytes <= HOT_FULL_DATASET_CACHE_MAX_BYTES {
                    let mut cached_bytes = vec![0u8; total_bytes];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            result_ptr,
                            cached_bytes.as_mut_ptr(),
                            total_bytes,
                        );
                    }
                    let _ = self.full_dataset_bytes.set(Arc::new(cached_bytes));
                }

                let mut result_shape = Vec::with_capacity(shape.len());
                for &dim in shape {
                    result_shape.push(checked_usize(dim, "dataset dimension")?);
                }
                let result_values = assume_init_vec(result_values);
                return ArrayD::from_shape_vec(IxDyn(&result_shape), result_values)
                    .map_err(|e| Error::InvalidData(format!("array shape error: {e}")));
            }

            let mut flat_data = vec![MaybeUninit::<u8>::uninit(); total_bytes];
            let flat_ptr = flat_data.as_mut_ptr() as *mut u8;
            let flat_len = flat_data.len();

            for entry in &entries {
                let chunk_data =
                    self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)?;
                unsafe {
                    copy_chunk_to_flat_with_strides_ptr(
                        &chunk_data,
                        FlatBufferPtr {
                            ptr: flat_ptr,
                            len: flat_len,
                        },
                        ChunkCopyLayout {
                            chunk_offsets: &entry.offsets,
                            chunk_shape: &chunk_shape,
                            dataset_shape: shape,
                            dataset_strides: &dataset_strides,
                            chunk_strides: &chunk_strides,
                            elem_size,
                        },
                    );
                }
            }

            let flat_data = assume_init_u8_vec(flat_data);
            if total_bytes <= HOT_FULL_DATASET_CACHE_MAX_BYTES {
                let _ = self.full_dataset_bytes.set(Arc::new(flat_data.clone()));
            }
            return self.decode_raw_data::<T>(&flat_data);
        }

        let mut flat_data = self.make_output_buffer(total_bytes);
        for entry in &entries {
            let chunk_data = self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)?;
            copy_chunk_to_flat_with_strides(
                &chunk_data,
                &mut flat_data,
                ChunkCopyLayout {
                    chunk_offsets: &entry.offsets,
                    chunk_shape: &chunk_shape,
                    dataset_shape: shape,
                    dataset_strides: &dataset_strides,
                    chunk_strides: &chunk_strides,
                    elem_size,
                },
            );
        }

        self.decode_raw_data::<T>(&flat_data)
    }

    fn read_chunked_bytes(
        &self,
        index_address: u64,
        chunk_dims: &[u32],
        chunk_indexing: Option<&ChunkIndexing>,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        if Cursor::is_undefined_offset(index_address, self.offset_size()) {
            return Ok(self.make_output_buffer(total_bytes));
        }

        let ndim = self.ndim();
        let shape = &self.dataspace.dims;
        let elem_size = dtype_element_size(&self.datatype);
        let chunk_shape: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
        let dataset_strides = row_major_strides(shape, "dataset stride")?;
        let chunk_strides = row_major_strides(&chunk_shape, "chunk stride")?;

        let entries = self.collect_chunk_entries(
            index_address,
            chunk_dims,
            chunk_indexing,
            ChunkEntrySelection {
                shape,
                ndim,
                elem_size,
                chunk_bounds: None,
            },
        )?;

        let full_chunk_coverage = entries.len() == full_dataset_chunk_count(shape, &chunk_shape)?;
        if full_chunk_coverage && total_bytes <= HOT_FULL_DATASET_CACHE_MAX_BYTES {
            if let Some(cached_bytes) = self.full_dataset_bytes.get() {
                return Ok(cached_bytes.as_ref().clone());
            }
        }

        let mut flat_data = self.make_output_buffer(total_bytes);
        for entry in &entries {
            let chunk_data = self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)?;
            copy_chunk_to_flat_with_strides(
                &chunk_data,
                &mut flat_data,
                ChunkCopyLayout {
                    chunk_offsets: &entry.offsets,
                    chunk_shape: &chunk_shape,
                    dataset_shape: shape,
                    dataset_strides: &dataset_strides,
                    chunk_strides: &chunk_strides,
                    elem_size,
                },
            );
        }

        if full_chunk_coverage && total_bytes <= HOT_FULL_DATASET_CACHE_MAX_BYTES {
            let _ = self.full_dataset_bytes.set(Arc::new(flat_data.clone()));
        }

        Ok(flat_data)
    }

    #[cfg(feature = "rayon")]
    fn read_chunked_parallel<T: H5Type>(
        &self,
        index_address: u64,
        chunk_dims: &[u32],
        _element_size: u32,
        chunk_indexing: Option<&ChunkIndexing>,
    ) -> Result<ArrayD<T>> {
        if Cursor::is_undefined_offset(index_address, self.offset_size()) {
            return self.make_fill_array::<T>();
        }

        let ndim = self.ndim();
        let shape = &self.dataspace.dims;
        let elem_size = dtype_element_size(&self.datatype);
        let chunk_shape: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
        let dataset_strides = row_major_strides(shape, "dataset stride")?;
        let chunk_strides = row_major_strides(&chunk_shape, "chunk stride")?;
        let total_elements = checked_usize(self.num_elements(), "dataset element count")?;
        let total_bytes = checked_mul_usize(total_elements, elem_size, "dataset size in bytes")?;

        let mut entries = self.collect_chunk_entries(
            index_address,
            chunk_dims,
            chunk_indexing,
            ChunkEntrySelection {
                shape,
                ndim,
                elem_size,
                chunk_bounds: None,
            },
        )?;

        // Dedup check: sort by output offsets and reject duplicates.
        // Two chunks claiming the same output offsets would cause data races
        // when writing into the flat buffer in parallel.
        entries.sort_by(|a, b| a.offsets.cmp(&b.offsets));
        for i in 1..entries.len() {
            if entries[i].offsets == entries[i - 1].offsets {
                return Err(Error::InvalidData(format!(
                    "duplicate chunk output offsets {:?} (addresses {:#x} and {:#x})",
                    entries[i].offsets,
                    entries[i - 1].address,
                    entries[i].address
                )));
            }
        }

        let full_chunk_coverage = entries.len() == full_dataset_chunk_count(shape, &chunk_shape)?;
        if full_chunk_coverage {
            if total_bytes <= HOT_FULL_DATASET_CACHE_MAX_BYTES {
                if let Some(cached_bytes) = self.full_dataset_bytes.get() {
                    return self.decode_raw_data::<T>(cached_bytes);
                }
            }
            if T::native_copy_compatible(&self.datatype) && std::mem::size_of::<T>() == elem_size {
                let mut result_values: Vec<MaybeUninit<T>> =
                    std::iter::repeat_with(MaybeUninit::<T>::uninit)
                        .take(total_elements)
                        .collect();
                let flat = FlatBufferPtr {
                    ptr: result_values.as_mut_ptr() as *mut u8,
                    len: checked_mul_usize(
                        result_values.len(),
                        std::mem::size_of::<T>(),
                        "typed dataset size in bytes",
                    )?,
                };

                entries
                    .par_iter()
                    .map(|entry| {
                        self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)
                            .map(|data| unsafe {
                                flat.copy_chunk(
                                    &data,
                                    ChunkCopyLayout {
                                        chunk_offsets: &entry.offsets,
                                        chunk_shape: &chunk_shape,
                                        dataset_shape: shape,
                                        dataset_strides: &dataset_strides,
                                        chunk_strides: &chunk_strides,
                                        elem_size,
                                    },
                                );
                            })
                    })
                    .collect::<std::result::Result<Vec<_>, Error>>()?;

                let mut result_shape = Vec::with_capacity(shape.len());
                for &dim in shape {
                    result_shape.push(checked_usize(dim, "dataset dimension")?);
                }
                if total_bytes <= HOT_FULL_DATASET_CACHE_MAX_BYTES {
                    let mut cached_bytes = vec![0u8; total_bytes];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            flat.ptr,
                            cached_bytes.as_mut_ptr(),
                            total_bytes,
                        );
                    }
                    let _ = self.full_dataset_bytes.set(Arc::new(cached_bytes));
                }
                let result_values = assume_init_vec(result_values);
                return ArrayD::from_shape_vec(IxDyn(&result_shape), result_values)
                    .map_err(|e| Error::InvalidData(format!("array shape error: {e}")));
            }

            let mut flat_data = vec![MaybeUninit::<u8>::uninit(); total_bytes];
            let flat = FlatBufferPtr {
                ptr: flat_data.as_mut_ptr() as *mut u8,
                len: flat_data.len(),
            };

            entries
                .par_iter()
                .map(|entry| {
                    self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)
                        .map(|data| unsafe {
                            flat.copy_chunk(
                                &data,
                                ChunkCopyLayout {
                                    chunk_offsets: &entry.offsets,
                                    chunk_shape: &chunk_shape,
                                    dataset_shape: shape,
                                    dataset_strides: &dataset_strides,
                                    chunk_strides: &chunk_strides,
                                    elem_size,
                                },
                            );
                        })
                })
                .collect::<std::result::Result<Vec<_>, Error>>()?;

            let flat_data = assume_init_u8_vec(flat_data);
            if total_bytes <= HOT_FULL_DATASET_CACHE_MAX_BYTES {
                let _ = self.full_dataset_bytes.set(Arc::new(flat_data.clone()));
            }
            return self.decode_raw_data::<T>(&flat_data);
        }

        let mut flat_data = self.make_output_buffer(total_bytes);
        let flat = FlatBufferPtr {
            ptr: flat_data.as_mut_ptr(),
            len: flat_data.len(),
        };

        entries
            .par_iter()
            .map(|entry| {
                self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)
                    .map(|data| unsafe {
                        flat.copy_chunk(
                            &data,
                            ChunkCopyLayout {
                                chunk_offsets: &entry.offsets,
                                chunk_shape: &chunk_shape,
                                dataset_shape: shape,
                                dataset_strides: &dataset_strides,
                                chunk_strides: &chunk_strides,
                                elem_size,
                            },
                        );
                    })
            })
            .collect::<std::result::Result<Vec<_>, Error>>()?;

        self.decode_raw_data::<T>(&flat_data)
    }

    /// Collect all chunk entries by dispatching on the chunk indexing type.
    ///
    /// Shared by `read_chunked` and `read_chunked_slice`.
    fn collect_chunk_entries(
        &self,
        index_address: u64,
        chunk_dims: &[u32],
        chunk_indexing: Option<&ChunkIndexing>,
        selection: ChunkEntrySelection<'_>,
    ) -> Result<Vec<chunk_index::ChunkEntry>> {
        if selection.chunk_bounds.is_none() {
            if let Some(cached) = self.full_chunk_entries.get() {
                return Ok((**cached).clone());
            }
        }

        let cache_key =
            selection
                .chunk_bounds
                .map(|(first_chunk, last_chunk)| ChunkEntryCacheKey {
                    index_address,
                    first_chunk: SmallVec::from_slice(first_chunk),
                    last_chunk: SmallVec::from_slice(last_chunk),
                });

        if let Some(ref key) = cache_key {
            let mut cache = self.chunk_entry_cache.lock();
            if let Some(cached) = cache.get(key) {
                return Ok((**cached).clone());
            }
        }

        let entries = match chunk_indexing {
            None => {
                // V1-V3: B-tree v1 chunk indexing
                self.collect_btree_v1_entries(
                    index_address,
                    selection.ndim,
                    chunk_dims,
                    selection.chunk_bounds,
                )
            }
            Some(ChunkIndexing::SingleChunk {
                filtered_size,
                filters,
            }) => Ok(vec![chunk_index::single_chunk_entry(
                index_address,
                *filtered_size,
                *filters,
                selection.ndim,
            )]),
            Some(ChunkIndexing::BTreeV2) => chunk_index::collect_v2_chunk_entries_storage(
                self.context.storage.as_ref(),
                index_address,
                self.offset_size(),
                self.length_size(),
                selection.ndim as u32,
                chunk_dims,
                selection.chunk_bounds,
            ),
            Some(ChunkIndexing::Implicit) => Ok(chunk_index::collect_implicit_chunk_entries(
                index_address,
                selection.shape,
                chunk_dims,
                selection.elem_size,
                selection.chunk_bounds,
            )),
            Some(ChunkIndexing::FixedArray { .. }) => {
                crate::fixed_array::collect_fixed_array_chunk_entries_storage(
                    self.context.storage.as_ref(),
                    index_address,
                    self.offset_size(),
                    self.length_size(),
                    selection.shape,
                    chunk_dims,
                    selection.chunk_bounds,
                )
            }
            Some(ChunkIndexing::ExtensibleArray { .. }) => {
                crate::extensible_array::collect_extensible_array_chunk_entries_storage(
                    self.context.storage.as_ref(),
                    index_address,
                    self.offset_size(),
                    self.length_size(),
                    selection.shape,
                    chunk_dims,
                    selection.chunk_bounds,
                )
            }
        }?;

        if let Some(key) = cache_key {
            let mut cache = self.chunk_entry_cache.lock();
            cache.put(key, Arc::new(entries.clone()));
        } else {
            let _ = self.full_chunk_entries.set(Arc::new(entries.clone()));
        }

        Ok(entries)
    }

    /// Collect chunk entries from a B-tree v1 index.
    fn collect_btree_v1_entries(
        &self,
        btree_address: u64,
        ndim: usize,
        chunk_dims: &[u32],
        chunk_bounds: Option<(&[u64], &[u64])>,
    ) -> Result<Vec<chunk_index::ChunkEntry>> {
        let leaves = crate::btree_v1::collect_btree_v1_leaves_storage(
            self.context.storage.as_ref(),
            btree_address,
            self.offset_size(),
            self.length_size(),
            Some(ndim as u32),
            chunk_dims,
            chunk_bounds,
        )?;

        let mut entries = Vec::with_capacity(leaves.len());
        for (key, chunk_addr) in &leaves {
            match key {
                crate::btree_v1::BTreeV1Key::RawData {
                    chunk_size,
                    filter_mask,
                    offsets,
                } => {
                    entries.push(chunk_index::ChunkEntry {
                        address: *chunk_addr,
                        size: *chunk_size as u64,
                        filter_mask: *filter_mask,
                        offsets: offsets[..ndim].to_vec(),
                    });
                }
                _ => {
                    return Err(Error::InvalidData(
                        "expected raw data key in chunk B-tree".into(),
                    ))
                }
            }
        }
        Ok(entries)
    }

    fn load_chunk_data(
        &self,
        entry: &chunk_index::ChunkEntry,
        dataset_addr: u64,
        chunk_shape: &[u64],
        elem_size: usize,
    ) -> Result<Arc<Vec<u8>>> {
        let cache_key = ChunkKey {
            dataset_addr,
            chunk_offsets: smallvec::SmallVec::from_slice(&entry.offsets),
        };

        self.chunk_cache.get_or_insert_with(cache_key, || {
            let size = if entry.size > 0 {
                checked_usize(entry.size, "encoded chunk size")?
            } else {
                let chunk_elements =
                    checked_shape_elements_usize(chunk_shape, "chunk element count")?;
                checked_mul_usize(chunk_elements, elem_size, "chunk byte size")?
            };
            let raw = self.context.read_range(entry.address, size)?;

            if let Some(ref pipeline) = self.filters {
                filters::apply_pipeline(
                    raw.as_ref(),
                    &pipeline.filters,
                    entry.filter_mask,
                    elem_size,
                    Some(&self.filter_registry),
                )
            } else {
                Ok(raw.to_vec())
            }
        })
    }

    /// Chunked slice: only read chunks that overlap the selection.
    ///
    /// Resolves each `SliceInfoElem` to concrete ranges, computes the chunk
    /// grid range per dimension, and only decompresses overlapping chunks.
    fn read_chunked_slice<T: H5Type>(
        &self,
        index_address: u64,
        chunk_dims: &[u32],
        _element_size: u32,
        chunk_indexing: Option<&ChunkIndexing>,
        _selection: &SliceInfo,
        resolved: &ResolvedSelection,
    ) -> Result<ArrayD<T>> {
        if resolved.result_elements == 0 {
            return self.make_fill_array_from_shape::<T>(0, &resolved.result_shape);
        }

        if Cursor::is_undefined_offset(index_address, self.offset_size()) {
            return self
                .make_fill_array_from_shape::<T>(resolved.result_elements, &resolved.result_shape);
        }

        let ndim = self.ndim();
        let shape = &self.dataspace.dims;
        let elem_size = dtype_element_size(&self.datatype);
        let chunk_shape: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
        let mut first_chunk = vec![0u64; ndim];
        let mut last_chunk = vec![0u64; ndim];
        for d in 0..ndim {
            let (first, last) = resolved.dims[d]
                .chunk_index_range(chunk_shape[d])
                .expect("zero-sized result handled above");
            first_chunk[d] = first;
            last_chunk[d] = last;
        }

        // Collect all chunk entries.
        let overlapping = self.collect_chunk_entries(
            index_address,
            chunk_dims,
            chunk_indexing,
            ChunkEntrySelection {
                shape,
                ndim,
                elem_size,
                chunk_bounds: Some((&first_chunk, &last_chunk)),
            },
        )?;

        let result_total_bytes = checked_mul_usize(
            resolved.result_elements,
            elem_size,
            "slice result size in bytes",
        )?;
        // Compute result strides (including collapsed dims — they have count=1).
        let result_dims = resolved.result_dims_with_collapsed();
        let mut result_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            result_strides[d] =
                checked_mul_usize(result_strides[d + 1], result_dims[d + 1], "result stride")?;
        }
        let mut chunk_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            chunk_strides[d] = checked_mul_usize(
                chunk_strides[d + 1],
                chunk_shape[d + 1] as usize,
                "chunk stride",
            )?;
        }
        let use_unit_stride_fast_path = resolved.is_unit_stride();
        let fully_covered_unit_stride = use_unit_stride_fast_path
            && overlapping.len() == expected_chunk_count(&first_chunk, &last_chunk)?;

        if fully_covered_unit_stride {
            if T::native_copy_compatible(&self.datatype) && std::mem::size_of::<T>() == elem_size {
                let mut result_values: Vec<MaybeUninit<T>> =
                    std::iter::repeat_with(MaybeUninit::<T>::uninit)
                        .take(resolved.result_elements)
                        .collect();
                let result_ptr = result_values.as_mut_ptr() as *mut u8;
                let result_len = checked_mul_usize(
                    result_values.len(),
                    std::mem::size_of::<T>(),
                    "typed slice result size in bytes",
                )?;

                for entry in &overlapping {
                    let chunk_data =
                        self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)?;

                    unsafe {
                        copy_unit_stride_chunk_overlap_ptr(
                            &chunk_data,
                            FlatBufferPtr {
                                ptr: result_ptr,
                                len: result_len,
                            },
                            UnitStrideCopyLayout {
                                chunk_offsets: &entry.offsets,
                                chunk_shape: &chunk_shape,
                                dataset_shape: shape,
                                resolved,
                                chunk_strides: &chunk_strides,
                                result_strides: &result_strides,
                                elem_size,
                            },
                        )?;
                    }
                }

                let result_values = assume_init_vec(result_values);
                return ArrayD::from_shape_vec(IxDyn(&resolved.result_shape), result_values)
                    .map_err(|e| Error::InvalidData(format!("array shape error: {e}")));
            }

            let mut result_buf = vec![MaybeUninit::<u8>::uninit(); result_total_bytes];
            let result_ptr = result_buf.as_mut_ptr() as *mut u8;
            let result_len = result_buf.len();

            for entry in &overlapping {
                let chunk_data =
                    self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)?;

                unsafe {
                    copy_unit_stride_chunk_overlap_ptr(
                        &chunk_data,
                        FlatBufferPtr {
                            ptr: result_ptr,
                            len: result_len,
                        },
                        UnitStrideCopyLayout {
                            chunk_offsets: &entry.offsets,
                            chunk_shape: &chunk_shape,
                            dataset_shape: shape,
                            resolved,
                            chunk_strides: &chunk_strides,
                            result_strides: &result_strides,
                            elem_size,
                        },
                    )?;
                }
            }

            let result_buf = assume_init_u8_vec(result_buf);
            return self.decode_buffer_with_shape::<T>(
                &result_buf,
                resolved.result_elements,
                &resolved.result_shape,
            );
        }

        let mut result_buf = self.make_output_buffer(result_total_bytes);

        // For each overlapping chunk: decompress and copy matching elements.
        for entry in &overlapping {
            let chunk_data = self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)?;

            if use_unit_stride_fast_path {
                copy_unit_stride_chunk_overlap(
                    &chunk_data,
                    &mut result_buf,
                    UnitStrideCopyLayout {
                        chunk_offsets: &entry.offsets,
                        chunk_shape: &chunk_shape,
                        dataset_shape: shape,
                        resolved,
                        chunk_strides: &chunk_strides,
                        result_strides: &result_strides,
                        elem_size,
                    },
                )?;
                continue;
            }

            // For each dimension, compute which elements within this chunk fall
            // within the selection.
            let mut dim_indices: Vec<Vec<(usize, usize)>> = Vec::with_capacity(ndim);
            for d in 0..ndim {
                let chunk_start = entry.offsets[d];
                let chunk_end = (chunk_start + chunk_shape[d]).min(shape[d]);
                let dim = &resolved.dims[d];
                let sel_start = dim.start;
                let sel_end = dim.end;
                let sel_step = dim.step;
                let mut indices = Vec::new();

                // Find first selected index >= chunk_start
                let first_sel = if sel_start >= chunk_start {
                    sel_start
                } else {
                    let steps_to_skip = (chunk_start - sel_start).div_ceil(sel_step);
                    sel_start + steps_to_skip * sel_step
                };

                let mut sel_idx = first_sel;
                while sel_idx < sel_end && sel_idx < chunk_end {
                    let chunk_local = checked_usize(sel_idx - chunk_start, "chunk-local index")?;
                    // Compute result-space index for this dimension.
                    let result_dim_idx =
                        checked_usize((sel_idx - dim.start) / sel_step, "result index")?;
                    indices.push((chunk_local, result_dim_idx));
                    sel_idx += sel_step;
                }

                dim_indices.push(indices);
            }

            // Iterate over the cartesian product of matching indices.
            copy_selected_elements(
                &chunk_data,
                &mut result_buf,
                &dim_indices,
                &chunk_strides,
                &result_strides,
                elem_size,
                ndim,
            );
        }

        self.decode_buffer_with_shape::<T>(
            &result_buf,
            resolved.result_elements,
            &resolved.result_shape,
        )
    }

    /// Parallel variant of `read_chunked_slice`: decompresses overlapping chunks
    /// in parallel using Rayon, then copies selected elements into the result buffer.
    ///
    /// Each chunk writes to a disjoint region of the result buffer (chunks don't
    /// overlap in output space), so this is safe to parallelize.
    #[cfg(feature = "rayon")]
    fn read_chunked_slice_parallel<T: H5Type>(
        &self,
        index_address: u64,
        chunk_dims: &[u32],
        _element_size: u32,
        chunk_indexing: Option<&ChunkIndexing>,
        _selection: &SliceInfo,
        resolved: &ResolvedSelection,
    ) -> Result<ArrayD<T>> {
        if resolved.result_elements == 0 {
            return self.make_fill_array_from_shape::<T>(0, &resolved.result_shape);
        }

        if Cursor::is_undefined_offset(index_address, self.offset_size()) {
            return self
                .make_fill_array_from_shape::<T>(resolved.result_elements, &resolved.result_shape);
        }

        let ndim = self.ndim();
        let shape = &self.dataspace.dims;
        let elem_size = dtype_element_size(&self.datatype);
        let chunk_shape: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
        let mut first_chunk = vec![0u64; ndim];
        let mut last_chunk = vec![0u64; ndim];
        for d in 0..ndim {
            let (first, last) = resolved.dims[d]
                .chunk_index_range(chunk_shape[d])
                .expect("zero-sized result handled above");
            first_chunk[d] = first;
            last_chunk[d] = last;
        }

        // Collect all chunk entries.
        let overlapping = self.collect_chunk_entries(
            index_address,
            chunk_dims,
            chunk_indexing,
            ChunkEntrySelection {
                shape,
                ndim,
                elem_size,
                chunk_bounds: Some((&first_chunk, &last_chunk)),
            },
        )?;

        // Allocate result buffer (raw bytes) initialized from fill value.
        let result_total_bytes = checked_mul_usize(
            resolved.result_elements,
            elem_size,
            "slice result size in bytes",
        )?;
        // Compute result strides (including collapsed dims — they have count=1).
        let result_dims = resolved.result_dims_with_collapsed();
        let mut result_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            result_strides[d] =
                checked_mul_usize(result_strides[d + 1], result_dims[d + 1], "result stride")?;
        }
        let mut chunk_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            chunk_strides[d] = checked_mul_usize(
                chunk_strides[d + 1],
                chunk_shape[d + 1] as usize,
                "chunk stride",
            )?;
        }
        let use_unit_stride_fast_path = resolved.is_unit_stride();
        let fully_covered_unit_stride = use_unit_stride_fast_path
            && overlapping.len() == expected_chunk_count(&first_chunk, &last_chunk)?;

        if fully_covered_unit_stride {
            if T::native_copy_compatible(&self.datatype) && std::mem::size_of::<T>() == elem_size {
                let mut result_values: Vec<MaybeUninit<T>> =
                    std::iter::repeat_with(MaybeUninit::<T>::uninit)
                        .take(resolved.result_elements)
                        .collect();
                let flat = FlatBufferPtr {
                    ptr: result_values.as_mut_ptr() as *mut u8,
                    len: checked_mul_usize(
                        result_values.len(),
                        std::mem::size_of::<T>(),
                        "typed slice result size in bytes",
                    )?,
                };

                overlapping
                    .par_iter()
                    .map(|entry| {
                        let chunk_data =
                            self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)?;

                        unsafe {
                            flat.copy_unit_stride_chunk_overlap(
                                &chunk_data,
                                UnitStrideCopyLayout {
                                    chunk_offsets: &entry.offsets,
                                    chunk_shape: &chunk_shape,
                                    dataset_shape: shape,
                                    resolved,
                                    chunk_strides: &chunk_strides,
                                    result_strides: &result_strides,
                                    elem_size,
                                },
                            )?;
                        }

                        Ok(())
                    })
                    .collect::<std::result::Result<Vec<_>, Error>>()?;

                let result_values = assume_init_vec(result_values);
                return ArrayD::from_shape_vec(IxDyn(&resolved.result_shape), result_values)
                    .map_err(|e| Error::InvalidData(format!("array shape error: {e}")));
            }

            let mut result_buf = vec![MaybeUninit::<u8>::uninit(); result_total_bytes];
            let flat = FlatBufferPtr {
                ptr: result_buf.as_mut_ptr() as *mut u8,
                len: result_buf.len(),
            };

            overlapping
                .par_iter()
                .map(|entry| {
                    let chunk_data =
                        self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)?;

                    unsafe {
                        flat.copy_unit_stride_chunk_overlap(
                            &chunk_data,
                            UnitStrideCopyLayout {
                                chunk_offsets: &entry.offsets,
                                chunk_shape: &chunk_shape,
                                dataset_shape: shape,
                                resolved,
                                chunk_strides: &chunk_strides,
                                result_strides: &result_strides,
                                elem_size,
                            },
                        )?;
                    }

                    Ok(())
                })
                .collect::<std::result::Result<Vec<_>, Error>>()?;

            let result_buf = assume_init_u8_vec(result_buf);
            return self.decode_buffer_with_shape::<T>(
                &result_buf,
                resolved.result_elements,
                &resolved.result_shape,
            );
        }

        let mut result_buf = self.make_output_buffer(result_total_bytes);

        let flat = FlatBufferPtr {
            ptr: result_buf.as_mut_ptr(),
            len: result_buf.len(),
        };

        overlapping
            .par_iter()
            .map(|entry| {
                let chunk_data =
                    self.load_chunk_data(entry, index_address, &chunk_shape, elem_size)?;

                if use_unit_stride_fast_path {
                    unsafe {
                        flat.copy_unit_stride_chunk_overlap(
                            &chunk_data,
                            UnitStrideCopyLayout {
                                chunk_offsets: &entry.offsets,
                                chunk_shape: &chunk_shape,
                                dataset_shape: shape,
                                resolved,
                                chunk_strides: &chunk_strides,
                                result_strides: &result_strides,
                                elem_size,
                            },
                        )?;
                    }
                    return Ok(());
                }

                // For each dimension, compute which elements within this chunk fall
                // within the selection.
                let mut dim_indices: Vec<Vec<(usize, usize)>> = Vec::with_capacity(ndim);
                for d in 0..ndim {
                    let chunk_start = entry.offsets[d];
                    let chunk_end = (chunk_start + chunk_shape[d]).min(shape[d]);
                    let dim = &resolved.dims[d];
                    let sel_start = dim.start;
                    let sel_end = dim.end;
                    let sel_step = dim.step;
                    let mut indices = Vec::new();

                    let first_sel = if sel_start >= chunk_start {
                        sel_start
                    } else {
                        let steps_to_skip = (chunk_start - sel_start).div_ceil(sel_step);
                        sel_start + steps_to_skip * sel_step
                    };

                    let mut sel_idx = first_sel;
                    while sel_idx < sel_end && sel_idx < chunk_end {
                        let chunk_local =
                            checked_usize(sel_idx - chunk_start, "chunk-local index")?;
                        let result_dim_idx =
                            checked_usize((sel_idx - dim.start) / sel_step, "result index")?;
                        indices.push((chunk_local, result_dim_idx));
                        sel_idx += sel_step;
                    }

                    dim_indices.push(indices);
                }

                // SAFETY: each chunk writes to disjoint output positions because
                // chunks occupy non-overlapping regions of the dataset grid and
                // the selection maps each dataset coordinate to a unique result index.
                unsafe {
                    flat.copy_selected(
                        &chunk_data,
                        &dim_indices,
                        &chunk_strides,
                        &result_strides,
                        elem_size,
                        ndim,
                    );
                }

                Ok(())
            })
            .collect::<std::result::Result<Vec<_>, Error>>()?;

        self.decode_buffer_with_shape::<T>(
            &result_buf,
            resolved.result_elements,
            &resolved.result_shape,
        )
    }

    fn read_contiguous_slice<T: H5Type>(
        &self,
        address: u64,
        size: u64,
        _selection: &SliceInfo,
        resolved: &ResolvedSelection,
    ) -> Result<ArrayD<T>> {
        if resolved.result_elements == 0 {
            return self.make_fill_array_from_shape::<T>(0, &resolved.result_shape);
        }

        if Cursor::is_undefined_offset(address, self.offset_size()) || size == 0 {
            return self
                .make_fill_array_from_shape::<T>(resolved.result_elements, &resolved.result_shape);
        }

        let shape = &self.dataspace.dims;
        let ndim = shape.len();
        let elem_size = dtype_element_size(&self.datatype);
        let result_total_bytes = checked_mul_usize(
            resolved.result_elements,
            elem_size,
            "contiguous slice result size in bytes",
        )?;
        let storage_len = checked_usize(size, "contiguous dataset size")?;
        let raw = self.context.read_range(address, storage_len)?;
        let dataset_strides = row_major_strides(shape, "contiguous dataset stride")?;
        let result_dims = resolved.result_dims_with_collapsed();
        let zero_offsets = vec![0u64; ndim];
        let mut result_strides = vec![1usize; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            result_strides[d] =
                checked_mul_usize(result_strides[d + 1], result_dims[d + 1], "result stride")?;
        }

        if resolved.is_unit_stride() {
            if T::native_copy_compatible(&self.datatype) && std::mem::size_of::<T>() == elem_size {
                let mut result_values: Vec<MaybeUninit<T>> =
                    std::iter::repeat_with(MaybeUninit::<T>::uninit)
                        .take(resolved.result_elements)
                        .collect();
                let flat = FlatBufferPtr {
                    ptr: result_values.as_mut_ptr() as *mut u8,
                    len: checked_mul_usize(
                        result_values.len(),
                        std::mem::size_of::<T>(),
                        "typed contiguous slice size in bytes",
                    )?,
                };

                unsafe {
                    copy_unit_stride_chunk_overlap_ptr(
                        raw.as_ref(),
                        flat,
                        UnitStrideCopyLayout {
                            chunk_offsets: &zero_offsets,
                            chunk_shape: shape,
                            dataset_shape: shape,
                            resolved,
                            chunk_strides: &dataset_strides,
                            result_strides: &result_strides,
                            elem_size,
                        },
                    )?;
                }

                let result_values = assume_init_vec(result_values);
                return ArrayD::from_shape_vec(IxDyn(&resolved.result_shape), result_values)
                    .map_err(|e| Error::InvalidData(format!("contiguous slice shape error: {e}")));
            }

            let mut result_buf = vec![MaybeUninit::<u8>::uninit(); result_total_bytes];
            unsafe {
                copy_unit_stride_chunk_overlap_ptr(
                    raw.as_ref(),
                    FlatBufferPtr {
                        ptr: result_buf.as_mut_ptr() as *mut u8,
                        len: result_buf.len(),
                    },
                    UnitStrideCopyLayout {
                        chunk_offsets: &zero_offsets,
                        chunk_shape: shape,
                        dataset_shape: shape,
                        resolved,
                        chunk_strides: &dataset_strides,
                        result_strides: &result_strides,
                        elem_size,
                    },
                )?;
            }

            let result_buf = assume_init_u8_vec(result_buf);
            return self.decode_buffer_with_shape::<T>(
                &result_buf,
                resolved.result_elements,
                &resolved.result_shape,
            );
        }

        let mut result_buf = self.make_output_buffer(result_total_bytes);
        let mut dim_indices: Vec<Vec<(usize, usize)>> = Vec::with_capacity(ndim);
        for dim in &resolved.dims {
            let mut indices = Vec::with_capacity(dim.count);
            let mut sel_idx = dim.start;
            while sel_idx < dim.end {
                let src_idx = checked_usize(sel_idx, "contiguous selection source index")?;
                let result_idx = checked_usize(
                    (sel_idx - dim.start) / dim.step,
                    "contiguous selection result index",
                )?;
                indices.push((src_idx, result_idx));
                sel_idx = sel_idx.saturating_add(dim.step);
            }
            dim_indices.push(indices);
        }

        copy_selected_elements(
            raw.as_ref(),
            &mut result_buf,
            &dim_indices,
            &dataset_strides,
            &result_strides,
            elem_size,
            ndim,
        );

        self.decode_buffer_with_shape::<T>(
            &result_buf,
            resolved.result_elements,
            &resolved.result_shape,
        )
    }

    fn read_compact_slice<T: H5Type>(
        &self,
        data: &[u8],
        selection: &SliceInfo,
    ) -> Result<ArrayD<T>> {
        let full = self.read_compact::<T>(data)?;
        slice_array(&full, selection, &self.dataspace.dims)
    }

    fn decode_buffer_with_shape<T: H5Type>(
        &self,
        raw: &[u8],
        n: usize,
        shape: &[usize],
    ) -> Result<ArrayD<T>> {
        let elem_size = dtype_element_size(&self.datatype);

        if let Some(elements) = T::decode_vec(raw, &self.datatype, n) {
            let elements = elements?;
            return ArrayD::from_shape_vec(IxDyn(shape), elements)
                .map_err(|e| Error::InvalidData(format!("array shape error: {e}")));
        }

        let mut elements = Vec::with_capacity(n);
        for i in 0..n {
            let start = checked_mul_usize(i, elem_size, "decoded element byte offset")?;
            let end = checked_mul_usize(i + 1, elem_size, "decoded element end offset")?;
            if end > raw.len() {
                // Pad with fill value or zeros if data is short.
                let padded = if end <= raw.len().saturating_add(elem_size) {
                    let mut buf = vec![0u8; elem_size];
                    let available = raw.len().saturating_sub(start);
                    if available > 0 {
                        buf[..available].copy_from_slice(&raw[start..start + available]);
                    }
                    T::from_bytes(&buf, &self.datatype)?
                } else {
                    T::from_bytes(&vec![0u8; elem_size], &self.datatype)?
                };
                elements.push(padded);
            } else {
                elements.push(T::from_bytes(&raw[start..end], &self.datatype)?);
            }
        }

        ArrayD::from_shape_vec(IxDyn(shape), elements)
            .map_err(|e| Error::InvalidData(format!("array shape error: {e}")))
    }

    fn decode_raw_data<T: H5Type>(&self, raw: &[u8]) -> Result<ArrayD<T>> {
        let n = checked_usize(self.num_elements(), "dataset element count")?;
        let mut shape = Vec::with_capacity(self.dataspace.dims.len());
        for &dim in &self.dataspace.dims {
            shape.push(checked_usize(dim, "dataset dimension")?);
        }
        self.decode_buffer_with_shape::<T>(raw, n, &shape)
    }

    fn make_fill_array<T: H5Type>(&self) -> Result<ArrayD<T>> {
        let n = checked_usize(self.num_elements(), "dataset element count")?;
        let mut shape = Vec::with_capacity(self.dataspace.dims.len());
        for &dim in &self.dataspace.dims {
            shape.push(checked_usize(dim, "dataset dimension")?);
        }
        self.make_fill_array_from_shape::<T>(n, &shape)
    }

    fn make_fill_array_from_shape<T: H5Type>(
        &self,
        element_count: usize,
        shape: &[usize],
    ) -> Result<ArrayD<T>> {
        let elem_size = dtype_element_size(&self.datatype);
        let total_bytes = checked_mul_usize(element_count, elem_size, "fill result size in bytes")?;
        let fill = self.make_output_buffer(total_bytes);
        self.decode_buffer_with_shape::<T>(&fill, element_count, shape)
    }

    fn make_output_buffer(&self, total_bytes: usize) -> Vec<u8> {
        if let Some(ref fv) = self.fill_value {
            if let Some(ref fill_bytes) = fv.value {
                let mut buf = vec![0u8; total_bytes];
                if !fill_bytes.is_empty() {
                    for chunk in buf.chunks_exact_mut(fill_bytes.len()) {
                        chunk.copy_from_slice(fill_bytes);
                    }
                }
                buf
            } else {
                vec![0u8; total_bytes]
            }
        } else {
            vec![0u8; total_bytes]
        }
    }

    fn normalize_raw_bytes(&self, raw: &[u8], total_bytes: usize) -> Vec<u8> {
        if raw.len() >= total_bytes {
            raw[..total_bytes].to_vec()
        } else {
            let mut normalized = self.make_output_buffer(total_bytes);
            normalized[..raw.len()].copy_from_slice(raw);
            normalized
        }
    }
}

fn attribute_from_message_storage(message: &AttributeMessage, context: &FileContext) -> Attribute {
    let raw_data = match &message.datatype {
        Datatype::VarLen { base }
            if matches!(base.as_ref(), Datatype::FixedPoint { size: 1, .. })
                && message.dataspace.num_elements() == 1 =>
        {
            resolve_vlen_bytes_storage(
                &message.raw_data,
                context.storage.as_ref(),
                context.superblock.offset_size,
                context.superblock.length_size,
            )
            .unwrap_or_else(|| message.raw_data.clone())
        }
        _ => message.raw_data.clone(),
    };

    Attribute {
        name: message.name.clone(),
        datatype: message.datatype.clone(),
        shape: match message.dataspace.dataspace_type {
            DataspaceType::Scalar => vec![],
            DataspaceType::Null => vec![0],
            DataspaceType::Simple => message.dataspace.dims.clone(),
        },
        raw_data,
    }
}

fn normalize_layout(layout: DataLayout, dataspace: &DataspaceMessage) -> DataLayout {
    match layout {
        DataLayout::Chunked {
            address,
            mut dims,
            mut element_size,
            chunk_indexing,
        } if dims.len() == dataspace.dims.len() + 1 => {
            if let Some(legacy_element_size) = dims.pop() {
                if element_size == 0 {
                    element_size = legacy_element_size;
                }
            }
            DataLayout::Chunked {
                address,
                dims,
                element_size,
                chunk_indexing,
            }
        }
        other => other,
    }
}

#[cfg(test)]
/// Copy a chunk's data into the flat output buffer at the correct position.
fn copy_chunk_to_flat(
    chunk_data: &[u8],
    flat: &mut [u8],
    chunk_offsets: &[u64],
    chunk_shape: &[u64],
    dataset_shape: &[u64],
    elem_size: usize,
) {
    let dataset_strides = row_major_strides(dataset_shape, "dataset stride")
        .expect("dataset strides should fit in usize");
    let chunk_strides =
        row_major_strides(chunk_shape, "chunk stride").expect("chunk strides should fit in usize");
    copy_chunk_to_flat_with_strides(
        chunk_data,
        flat,
        ChunkCopyLayout {
            chunk_offsets,
            chunk_shape,
            dataset_shape,
            dataset_strides: &dataset_strides,
            chunk_strides: &chunk_strides,
            elem_size,
        },
    );
}

fn copy_chunk_to_flat_with_strides(
    chunk_data: &[u8],
    flat: &mut [u8],
    layout: ChunkCopyLayout<'_>,
) {
    unsafe {
        copy_chunk_to_flat_with_strides_ptr(
            chunk_data,
            FlatBufferPtr {
                ptr: flat.as_mut_ptr(),
                len: flat.len(),
            },
            layout,
        );
    }
}

#[inline(always)]
unsafe fn copy_chunk_to_flat_with_strides_ptr(
    chunk_data: &[u8],
    flat: FlatBufferPtr,
    layout: ChunkCopyLayout<'_>,
) {
    let ndim = layout.dataset_shape.len();

    if ndim == 0 {
        let bytes = layout.elem_size.min(chunk_data.len()).min(flat.len);
        std::ptr::copy_nonoverlapping(chunk_data.as_ptr(), flat.ptr, bytes);
        return;
    }

    // Total elements in this chunk (clamped to dataset boundaries)
    let mut actual_chunk_shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let remaining = layout.dataset_shape[i] - layout.chunk_offsets[i];
        actual_chunk_shape.push(remaining.min(layout.chunk_shape[i]) as usize);
    }

    let row_elems = *actual_chunk_shape.last().unwrap_or(&1);
    let row_bytes = row_elems * layout.elem_size;
    let dataset_origin: usize = layout
        .chunk_offsets
        .iter()
        .enumerate()
        .map(|(d, offset)| *offset as usize * layout.dataset_strides[d])
        .sum();

    if ndim == 1 {
        let bytes = row_bytes.min(chunk_data.len());
        let dst_start = dataset_origin * layout.elem_size;
        let dst_end = dst_start + bytes;
        if dst_end <= flat.len {
            std::ptr::copy_nonoverlapping(chunk_data.as_ptr(), flat.ptr.add(dst_start), bytes);
        }
        return;
    }

    let outer_dims = &actual_chunk_shape[..ndim - 1];
    let total_rows: usize = outer_dims.iter().product();
    let mut outer_idx = vec![0usize; ndim - 1];

    for _ in 0..total_rows {
        let mut chunk_row = 0usize;
        let mut dataset_row = dataset_origin;
        for (d, outer) in outer_idx.iter().copied().enumerate() {
            chunk_row += outer * layout.chunk_strides[d];
            dataset_row += outer * layout.dataset_strides[d];
        }

        let src_start = chunk_row * layout.elem_size;
        let dst_start = dataset_row * layout.elem_size;
        let src_end = src_start + row_bytes;
        let dst_end = dst_start + row_bytes;
        if src_end <= chunk_data.len() && dst_end <= flat.len {
            std::ptr::copy_nonoverlapping(
                chunk_data.as_ptr().add(src_start),
                flat.ptr.add(dst_start),
                row_bytes,
            );
        }

        let mut carry = true;
        for d in (0..outer_idx.len()).rev() {
            if carry {
                outer_idx[d] += 1;
                if outer_idx[d] < outer_dims[d] {
                    carry = false;
                } else {
                    outer_idx[d] = 0;
                }
            }
        }
    }
}

fn checked_product_usize(values: &[usize], context: &str) -> Result<usize> {
    let mut product = 1usize;
    for &value in values {
        product = checked_mul_usize(product, value, context)?;
    }
    Ok(product)
}

fn unit_stride_chunk_overlap_plan(
    chunk_offsets: &[u64],
    chunk_shape: &[u64],
    dataset_shape: &[u64],
    resolved: &ResolvedSelection,
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let ndim = dataset_shape.len();
    let mut overlap_counts = Vec::with_capacity(ndim);
    let mut chunk_local_start = Vec::with_capacity(ndim);
    let mut result_start = Vec::with_capacity(ndim);

    for d in 0..ndim {
        let chunk_start = chunk_offsets[d];
        let chunk_end = (chunk_start + chunk_shape[d]).min(dataset_shape[d]);
        let dim = &resolved.dims[d];
        let overlap_start = chunk_start.max(dim.start);
        let overlap_end = chunk_end.min(dim.end);
        if overlap_start >= overlap_end {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        overlap_counts.push(checked_usize(
            overlap_end - overlap_start,
            "chunk overlap size",
        )?);
        chunk_local_start.push(checked_usize(
            overlap_start - chunk_start,
            "chunk overlap start",
        )?);
        result_start.push(checked_usize(
            overlap_start - dim.start,
            "slice result overlap start",
        )?);
    }

    Ok((overlap_counts, chunk_local_start, result_start))
}

#[inline(always)]
fn copy_unit_stride_chunk_overlap(
    chunk_data: &[u8],
    result_buf: &mut [u8],
    layout: UnitStrideCopyLayout<'_>,
) -> Result<()> {
    unsafe {
        copy_unit_stride_chunk_overlap_ptr(
            chunk_data,
            FlatBufferPtr {
                ptr: result_buf.as_mut_ptr(),
                len: result_buf.len(),
            },
            layout,
        )
    }
}

/// Copy a unit-step rectangular overlap from a chunk into the result buffer.
///
/// This is the hot path for contiguous hyperslab reads over chunked datasets:
/// rather than copying one element at a time, it copies contiguous runs along
/// the innermost dimension with a single memcpy per output row.
///
/// # Safety
///
/// The caller must guarantee that `[result_ptr .. result_ptr + result_len)` is
/// valid for writes. Concurrent callers must write to disjoint byte ranges.
#[inline(always)]
unsafe fn copy_unit_stride_chunk_overlap_ptr(
    chunk_data: &[u8],
    result: FlatBufferPtr,
    layout: UnitStrideCopyLayout<'_>,
) -> Result<()> {
    let ndim = layout.dataset_shape.len();

    if ndim == 0 {
        let bytes = layout.elem_size.min(chunk_data.len()).min(result.len);
        std::ptr::copy_nonoverlapping(chunk_data.as_ptr(), result.ptr, bytes);
        return Ok(());
    }

    let (overlap_counts, chunk_local_start, result_start) = unit_stride_chunk_overlap_plan(
        layout.chunk_offsets,
        layout.chunk_shape,
        layout.dataset_shape,
        layout.resolved,
    )?;
    if overlap_counts.is_empty() {
        return Ok(());
    }

    let row_elems = *overlap_counts.last().unwrap_or(&1);
    let row_bytes = checked_mul_usize(row_elems, layout.elem_size, "unit-stride slice row bytes")?;

    let mut chunk_origin = 0usize;
    let mut result_origin = 0usize;
    for d in 0..ndim {
        let chunk_term = checked_mul_usize(
            chunk_local_start[d],
            layout.chunk_strides[d],
            "chunk overlap origin",
        )?;
        let result_term = checked_mul_usize(
            result_start[d],
            layout.result_strides[d],
            "slice result origin",
        )?;
        chunk_origin = checked_add_usize(chunk_origin, chunk_term, "chunk overlap origin")?;
        result_origin = checked_add_usize(result_origin, result_term, "slice result origin")?;
    }

    if ndim == 1 {
        let src_start = chunk_origin * layout.elem_size;
        let dst_start = result_origin * layout.elem_size;
        let src_end = src_start + row_bytes;
        let dst_end = dst_start + row_bytes;
        if src_end <= chunk_data.len() && dst_end <= result.len {
            std::ptr::copy_nonoverlapping(
                chunk_data.as_ptr().add(src_start),
                result.ptr.add(dst_start),
                row_bytes,
            );
        }
        return Ok(());
    }

    let outer_counts = &overlap_counts[..ndim - 1];
    let total_rows = checked_product_usize(outer_counts, "unit-stride slice row count")?;
    let mut outer_idx = vec![0usize; ndim - 1];

    for _ in 0..total_rows {
        let mut chunk_row = chunk_origin;
        let mut result_row = result_origin;
        for (d, outer) in outer_idx.iter().copied().enumerate() {
            chunk_row += outer * layout.chunk_strides[d];
            result_row += outer * layout.result_strides[d];
        }

        let src_start = chunk_row * layout.elem_size;
        let dst_start = result_row * layout.elem_size;
        let src_end = src_start + row_bytes;
        let dst_end = dst_start + row_bytes;
        if src_end <= chunk_data.len() && dst_end <= result.len {
            std::ptr::copy_nonoverlapping(
                chunk_data.as_ptr().add(src_start),
                result.ptr.add(dst_start),
                row_bytes,
            );
        }

        let mut carry = true;
        for d in (0..outer_idx.len()).rev() {
            if carry {
                outer_idx[d] += 1;
                if outer_idx[d] < outer_counts[d] {
                    carry = false;
                } else {
                    outer_idx[d] = 0;
                }
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Copy selected elements from a chunk into the result buffer.
///
/// `dim_indices[d]` is a list of `(chunk_local_idx, result_dim_idx)` pairs for dimension `d`.
#[inline(always)]
fn copy_selected_elements(
    chunk_data: &[u8],
    result_buf: &mut [u8],
    dim_indices: &[Vec<(usize, usize)>],
    chunk_strides: &[usize],
    result_strides: &[usize],
    elem_size: usize,
    ndim: usize,
) {
    // Check for empty selection
    if dim_indices.iter().any(|v| v.is_empty()) {
        return;
    }

    // Recursive cartesian-product iteration, but unrolled iteratively.
    let total: usize = dim_indices.iter().map(|v| v.len()).product();
    let mut counters = vec![0usize; ndim];

    for _ in 0..total {
        let mut chunk_flat = 0;
        let mut result_flat = 0;
        for d in 0..ndim {
            let (cl, ri) = dim_indices[d][counters[d]];
            chunk_flat += cl * chunk_strides[d];
            result_flat += ri * result_strides[d];
        }

        let src_start = chunk_flat * elem_size;
        let dst_start = result_flat * elem_size;
        let src_end = src_start + elem_size;
        let dst_end = dst_start + elem_size;

        if src_end <= chunk_data.len() && dst_end <= result_buf.len() {
            result_buf[dst_start..dst_end].copy_from_slice(&chunk_data[src_start..src_end]);
        }

        // Increment counters (row-major)
        let mut carry = true;
        for d in (0..ndim).rev() {
            if carry {
                counters[d] += 1;
                if counters[d] < dim_indices[d].len() {
                    carry = false;
                } else {
                    counters[d] = 0;
                }
            }
        }
    }
}

/// Copy selected elements from a chunk into a raw output pointer.
///
/// This is the pointer-based variant of `copy_selected_elements`, suitable for
/// parallel use where multiple threads write to disjoint regions of the same buffer.
///
/// # Safety
///
/// The caller must guarantee that no two concurrent calls write to the same
/// byte range within `[result_ptr .. result_ptr + result_len)`.
#[cfg(feature = "rayon")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn copy_selected_elements_ptr(
    chunk_data: &[u8],
    result_ptr: *mut u8,
    result_len: usize,
    dim_indices: &[Vec<(usize, usize)>],
    chunk_strides: &[usize],
    result_strides: &[usize],
    elem_size: usize,
    ndim: usize,
) {
    if dim_indices.iter().any(|v| v.is_empty()) {
        return;
    }

    let total: usize = dim_indices.iter().map(|v| v.len()).product();
    let mut counters = vec![0usize; ndim];

    for _ in 0..total {
        let mut chunk_flat = 0;
        let mut result_flat = 0;
        for d in 0..ndim {
            let (cl, ri) = dim_indices[d][counters[d]];
            chunk_flat += cl * chunk_strides[d];
            result_flat += ri * result_strides[d];
        }

        let src_start = chunk_flat * elem_size;
        let dst_start = result_flat * elem_size;
        let src_end = src_start + elem_size;
        let dst_end = dst_start + elem_size;

        if src_end <= chunk_data.len() && dst_end <= result_len {
            std::ptr::copy_nonoverlapping(
                chunk_data.as_ptr().add(src_start),
                result_ptr.add(dst_start),
                elem_size,
            );
        }

        let mut carry = true;
        for d in (0..ndim).rev() {
            if carry {
                counters[d] += 1;
                if counters[d] < dim_indices[d].len() {
                    carry = false;
                } else {
                    counters[d] = 0;
                }
            }
        }
    }
}

/// Slice an ndarray according to a SliceInfo selection.
fn slice_array<T: H5Type + Clone>(
    array: &ArrayD<T>,
    selection: &SliceInfo,
    shape: &[u64],
) -> Result<ArrayD<T>> {
    // Build result shape
    let mut result_shape = Vec::new();

    for (i, sel) in selection.selections.iter().enumerate() {
        let dim_size = shape[i];
        match sel {
            SliceInfoElem::Index(idx) => {
                if *idx >= dim_size {
                    return Err(Error::SliceOutOfBounds {
                        dim: i,
                        index: *idx,
                        size: dim_size,
                    });
                }
                // Don't add to result_shape — this dimension is collapsed
            }
            SliceInfoElem::Slice { start, end, step } => {
                let dim_size = checked_usize(dim_size, "slice dimension size")?;
                let actual_end = if *end == u64::MAX {
                    dim_size
                } else {
                    checked_usize(*end, "slice end")?.min(dim_size)
                };
                let actual_start = checked_usize(*start, "slice start")?;
                let actual_step = checked_usize(*step, "slice step")?;
                if actual_step == 0 {
                    return Err(Error::InvalidData("slice step cannot be 0".into()));
                }
                if actual_start > dim_size {
                    return Err(Error::SliceOutOfBounds {
                        dim: i,
                        index: *start,
                        size: shape[i],
                    });
                }
                let n = (actual_end - actual_start).div_ceil(actual_step);
                result_shape.push(n);
            }
        }
    }

    // Extract elements manually (ndarray's slicing API is complex with dynamic dims)
    let ndim = shape.len();
    let total = checked_product_usize(&result_shape, "slice result element count")?;
    let mut elements = Vec::with_capacity(total);

    // Generate all indices in the result
    let mut result_idx = vec![0usize; result_shape.len()];

    for _ in 0..total {
        // Map result index to source index
        let mut src_idx = Vec::with_capacity(ndim);
        let mut ri = 0;
        for sel in selection.selections.iter() {
            match sel {
                SliceInfoElem::Index(idx) => {
                    src_idx.push(checked_usize(*idx, "slice source index")?);
                }
                SliceInfoElem::Slice { start, step, .. } => {
                    let start = checked_usize(*start, "slice start")?;
                    let step = checked_usize(*step, "slice step")?;
                    let offset =
                        checked_mul_usize(result_idx[ri], step, "slice source index offset")?;
                    src_idx.push(checked_add_usize(start, offset, "slice source index")?);
                    ri += 1;
                }
            }
        }

        elements.push(array[IxDyn(&src_idx)].clone());

        // Increment result index
        if !result_shape.is_empty() {
            let mut carry = true;
            for d in (0..result_shape.len()).rev() {
                if carry {
                    result_idx[d] += 1;
                    if result_idx[d] < result_shape[d] {
                        carry = false;
                    } else {
                        result_idx[d] = 0;
                    }
                }
            }
        }
    }

    ArrayD::from_shape_vec(IxDyn(&result_shape), elements)
        .map_err(|e| Error::InvalidData(format!("slice shape error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_info_all() {
        let s = SliceInfo::all(3);
        assert_eq!(s.selections.len(), 3);
    }

    #[test]
    fn test_copy_chunk_1d() {
        let chunk_data = vec![1u8, 2, 3, 4]; // 4 elements of 1 byte each
        let mut flat = vec![0u8; 8];
        let chunk_offsets = vec![2u64]; // starts at index 2
        let chunk_shape = vec![4u64];
        let dataset_shape = vec![8u64];

        copy_chunk_to_flat(
            &chunk_data,
            &mut flat,
            &chunk_offsets,
            &chunk_shape,
            &dataset_shape,
            1,
        );
        assert_eq!(flat, vec![0, 0, 1, 2, 3, 4, 0, 0]);
    }

    #[test]
    fn test_copy_chunk_2d_rowwise() {
        let chunk_data = vec![1u8, 2, 3, 4, 5, 6];
        let mut flat = vec![0u8; 16];
        let chunk_offsets = vec![1u64, 1u64];
        let chunk_shape = vec![2u64, 3u64];
        let dataset_shape = vec![4u64, 4u64];

        copy_chunk_to_flat(
            &chunk_data,
            &mut flat,
            &chunk_offsets,
            &chunk_shape,
            &dataset_shape,
            1,
        );

        assert_eq!(flat, vec![0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 0, 0, 0,]);
    }

    #[test]
    fn test_copy_unit_stride_chunk_overlap_2d_partial() {
        let chunk_data: Vec<u8> = (1..=16).collect();
        let mut result = vec![0u8; 6];
        let chunk_offsets = vec![0u64, 0u64];
        let chunk_shape = vec![4u64, 4u64];
        let dataset_shape = vec![4u64, 4u64];
        let resolved = ResolvedSelection {
            dims: vec![
                ResolvedSelectionDim {
                    start: 1,
                    end: 3,
                    step: 1,
                    count: 2,
                },
                ResolvedSelectionDim {
                    start: 1,
                    end: 4,
                    step: 1,
                    count: 3,
                },
            ],
            result_shape: vec![2, 3],
            result_elements: 6,
        };
        let chunk_strides = vec![4usize, 1usize];
        let result_strides = vec![3usize, 1usize];

        copy_unit_stride_chunk_overlap(
            &chunk_data,
            &mut result,
            UnitStrideCopyLayout {
                chunk_offsets: &chunk_offsets,
                chunk_shape: &chunk_shape,
                dataset_shape: &dataset_shape,
                resolved: &resolved,
                chunk_strides: &chunk_strides,
                result_strides: &result_strides,
                elem_size: 1,
            },
        )
        .unwrap();

        assert_eq!(result, vec![6, 7, 8, 10, 11, 12]);
    }
}
