//! Writing HDF5 files.
//!
//! [`Writer`] builds a file in memory: groups, attributes and datasets
//! (contiguous, compact, or chunked with the shuffle and deflate filters),
//! plus the object references and variable-length data that netCDF-4
//! dimension scales need ([`Data::ObjectRefLists`],
//! [`Data::DimensionScaleRefs`]). [`Writer::finish`] lays the file out and
//! returns its bytes.
//!
//! The output is the HDF5 1.8 file format that every HDF5 library since
//! 1.8.0 reads (h5py, netCDF-C, the C++ API LROSE uses):
//!
//! - a version 2 superblock (8-byte offsets and lengths);
//! - version 2 object headers with lookup3 checksums, attribute creation
//!   order tracked, one header block per object;
//! - new-style groups: link info, group info and one hard link message per
//!   child in the header (compact storage), link creation order tracked so
//!   readers list children in insertion order;
//! - compact attributes (version 3 attribute messages, at most 64 KiB each);
//! - datasets with a version 3 layout message: contiguous, compact, or
//!   chunked with a version 1 B-tree chunk index (every reader of layout
//!   version 3 supports it); a version 2 filter pipeline (shuffle, deflate);
//!   a version 3 fill value message;
//! - variable-length strings and reference sequences in global heap
//!   collections of at least 4 KiB.
//!
//! Everything is little-endian. Writing is deterministic: the same calls
//! give the same bytes (no timestamps are stored).
//!
//! ```
//! use recast_radar_hdf5::write::{Data, Layout, NewDataset, Value, Writer};
//!
//! # fn main() -> Result<(), recast_radar_hdf5::write::WriteError> {
//! let mut writer = Writer::new();
//! let root = writer.root();
//! writer.add_attribute(root, "Conventions", Value::text_nul("ODIM_H5/V2_4"))?;
//! let group = writer.add_group(root, "dataset1")?;
//! let data = Value::array(Data::U8(vec![0; 360 * 250]), vec![360, 250]);
//! writer.add_dataset(group, "data", NewDataset::new(data, Layout::chunked(vec![360, 250], Some(6))))?;
//! let bytes = writer.finish()?;
//! let file = recast_radar_hdf5::H5File::open(&bytes).map_err(|e| {
//!     recast_radar_hdf5::write::WriteError::Invalid(e.to_string())
//! })?;
//! assert_eq!(file.dataset("/dataset1/data").map(|d| d.dims).ok(), Some(vec![360, 250]));
//! # Ok(())
//! # }
//! ```

mod chunked;
mod encode;
pub mod netcdf4;

use thiserror::Error;

pub use crate::datatype::{CharSet, StringPadding};

use encode::{AllocTime, Message, SUPERBLOCK_SIZE, UNDEF};

/// Errors from building or laying out a file.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WriteError {
    /// A link name HDF5 cannot store (empty, `.`, or containing `/` or NUL).
    #[error("invalid HDF5 link name {0:?}")]
    InvalidName(String),
    /// An attribute name HDF5 cannot store (empty or containing NUL).
    #[error("invalid HDF5 attribute name {0:?}")]
    InvalidAttributeName(String),
    /// A group already has a link, or an object an attribute, of this name.
    #[error("duplicate name {0:?}")]
    Duplicate(String),
    /// An [`ObjectId`] that is not this writer's, or a dataset used as a
    /// group.
    #[error("object {0} is not a group of this writer")]
    NotAGroup(usize),
    /// An [`ObjectId`] that is not this writer's.
    #[error("object {0} does not exist in this writer")]
    UnknownObject(usize),
    /// Values that do not match their shape, or another inconsistent value.
    #[error("invalid value: {0}")]
    Invalid(String),
    /// A structure larger than the format (or this writer) can store: an
    /// attribute message over 64 KiB, a chunk over 4 GiB, more than 65,535
    /// links or attributes on one object.
    #[error("too large: {0}")]
    TooLarge(String),
    /// A combination this writer does not implement (for example chunked
    /// variable-length data).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The deflate encoder failed.
    #[error("deflate failed: {0}")]
    Compression(String),
}

/// A group or dataset of a [`Writer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObjectId(usize);

/// The elements of an attribute or dataset, row-major, in their stored
/// type.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Data {
    /// 8-bit signed integers.
    I8(Vec<i8>),
    /// 8-bit unsigned integers.
    U8(Vec<u8>),
    /// 16-bit signed integers.
    I16(Vec<i16>),
    /// 16-bit unsigned integers.
    U16(Vec<u16>),
    /// 32-bit signed integers.
    I32(Vec<i32>),
    /// 32-bit unsigned integers.
    U32(Vec<u32>),
    /// 64-bit signed integers.
    I64(Vec<i64>),
    /// 64-bit unsigned integers.
    U64(Vec<u64>),
    /// IEEE binary32.
    F32(Vec<f32>),
    /// IEEE binary64.
    F64(Vec<f64>),
    /// Fixed-length strings of `size` bytes each, back to back
    /// (`bytes.len()` is a multiple of `size`).
    FixedStrings {
        /// All strings, each exactly `size` bytes (padding included).
        bytes: Vec<u8>,
        /// Bytes per string (at least 1).
        size: usize,
        /// Padding convention.
        padding: StringPadding,
        /// Character set.
        charset: CharSet,
    },
    /// Variable-length strings (stored in the global heap).
    VarStrings {
        /// The strings.
        values: Vec<String>,
        /// Character set.
        charset: CharSet,
    },
    /// Object references to groups or datasets of the same writer.
    ObjectRefs(Vec<ObjectId>),
    /// Variable-length sequences of object references, one per element (the
    /// `DIMENSION_LIST` attribute of the dimension scale convention).
    ObjectRefLists(Vec<Vec<ObjectId>>),
    /// `(dataset, dimension)` pairs of the `REFERENCE_LIST` attribute of the
    /// dimension scale convention: which axis of which dataset a scale is
    /// attached to.
    DimensionScaleRefs(Vec<(ObjectId, i32)>),
    /// Booleans as h5py stores them: an enumeration of `int8` with the
    /// members `FALSE` (0) and `TRUE` (1).
    Bools(Vec<bool>),
    /// A compound of named members, each a fixed-size column of the same
    /// length (numbers or fixed-length strings), packed without padding in
    /// member order.
    Compound(Vec<(String, Data)>),
}

impl Data {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::I8(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::FixedStrings { bytes, size, .. } => bytes.len() / (*size).max(1),
            Self::VarStrings { values, .. } => values.len(),
            Self::ObjectRefs(v) => v.len(),
            Self::ObjectRefLists(v) => v.len(),
            Self::DimensionScaleRefs(v) => v.len(),
            Self::Bools(v) => v.len(),
            Self::Compound(members) => members.first().map_or(0, |(_, column)| column.len()),
        }
    }

    /// `true` when there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes per stored element.
    fn element_size(&self) -> usize {
        match self {
            Self::I8(_) | Self::U8(_) | Self::Bools(_) => 1,
            Self::I16(_) | Self::U16(_) => 2,
            Self::I32(_) | Self::U32(_) | Self::F32(_) => 4,
            Self::I64(_) | Self::U64(_) | Self::F64(_) | Self::ObjectRefs(_) => 8,
            Self::FixedStrings { size, .. } => *size,
            Self::VarStrings { .. } | Self::ObjectRefLists(_) | Self::DimensionScaleRefs(_) => 16,
            Self::Compound(members) => members
                .iter()
                .map(|(_, column)| column.element_size())
                .sum(),
        }
    }

    /// `true` for element types stored in the global heap or holding
    /// addresses (their bytes depend on the layout).
    fn is_addressed(&self) -> bool {
        match self {
            Self::VarStrings { .. }
            | Self::ObjectRefs(_)
            | Self::ObjectRefLists(_)
            | Self::DimensionScaleRefs(_) => true,
            Self::Compound(members) => members.iter().any(|(_, column)| column.is_addressed()),
            _ => false,
        }
    }
}

/// The dataspace of an attribute or dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Shape {
    /// One element, no dimensions.
    Scalar,
    /// No elements at all (an empty netCDF text attribute).
    Null,
    /// A simple dataspace with these dimensions.
    Simple(Vec<u64>),
}

impl Shape {
    fn element_count(&self) -> Option<usize> {
        match self {
            Self::Scalar => Some(1),
            Self::Null => Some(0),
            Self::Simple(dims) => dims.iter().try_fold(1usize, |product, dim| {
                usize::try_from(*dim)
                    .ok()
                    .and_then(|dim| product.checked_mul(dim))
            }),
        }
    }
}

/// An attribute or dataset value: elements and dataspace.
#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    /// The elements, row-major.
    pub data: Data,
    /// The dataspace.
    pub shape: Shape,
}

impl Value {
    /// A scalar value; `data` must hold exactly one element.
    pub fn scalar(data: Data) -> Self {
        Self {
            data,
            shape: Shape::Scalar,
        }
    }

    /// A one-dimensional value of all of `data`'s elements.
    pub fn vector(data: Data) -> Self {
        let len = data.len() as u64;
        Self {
            data,
            shape: Shape::Simple(vec![len]),
        }
    }

    /// An N-dimensional value.
    pub fn array(data: Data, dims: Vec<u64>) -> Self {
        Self {
            data,
            shape: Shape::Simple(dims),
        }
    }

    /// Text as netCDF-C stores a `char` attribute: a scalar NUL-terminated
    /// fixed-length ASCII (or UTF-8) string exactly as long as the text, and
    /// for empty text a null dataspace of a one-byte string.
    pub fn text(text: &str) -> Self {
        let charset = if text.is_ascii() {
            CharSet::Ascii
        } else {
            CharSet::Utf8
        };
        if text.is_empty() {
            return Self {
                data: Data::FixedStrings {
                    bytes: Vec::new(),
                    size: 1,
                    padding: StringPadding::NullTerminate,
                    charset,
                },
                shape: Shape::Null,
            };
        }
        Self::scalar(Data::FixedStrings {
            bytes: text.as_bytes().to_vec(),
            size: text.len(),
            padding: StringPadding::NullTerminate,
            charset,
        })
    }

    /// Text as a scalar fixed-length string with its terminating NUL stored
    /// (`len + 1` bytes), as ODIM_H5 writers store string attributes.
    pub fn text_nul(text: &str) -> Self {
        let charset = if text.is_ascii() {
            CharSet::Ascii
        } else {
            CharSet::Utf8
        };
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        let size = bytes.len();
        Self::scalar(Data::FixedStrings {
            bytes,
            size,
            padding: StringPadding::NullTerminate,
            charset,
        })
    }

    /// One variable-length UTF-8 string (a netCDF-4 `string` scalar).
    pub fn var_text(text: &str) -> Self {
        Self::scalar(Data::VarStrings {
            values: vec![text.to_owned()],
            charset: CharSet::Utf8,
        })
    }

    /// Variable-length UTF-8 strings, one-dimensional.
    pub fn var_texts(values: Vec<String>) -> Self {
        Self::vector(Data::VarStrings {
            values,
            charset: CharSet::Utf8,
        })
    }
}

/// How a dataset's elements are stored.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Layout {
    /// One contiguous block after the metadata.
    Contiguous,
    /// Inside the object header (at most about 64 KiB).
    Compact,
    /// Chunks of this shape (one entry per dimension, each at least 1),
    /// optionally shuffled and deflated (level 0-9).
    Chunked {
        /// Chunk extent per dimension.
        chunk: Vec<u64>,
        /// Byte shuffle before compression.
        shuffle: bool,
        /// zlib level when deflated.
        deflate: Option<u32>,
    },
    /// No storage allocated: every element reads as the fill value (a
    /// netCDF-4 dimension without a coordinate variable). The value's
    /// data must be empty; only its type and shape are used.
    Unallocated,
}

impl Layout {
    /// Chunks of `chunk`, deflated at `deflate` when given, not shuffled.
    pub fn chunked(chunk: Vec<u64>, deflate: Option<u32>) -> Self {
        Self::Chunked {
            chunk,
            shuffle: false,
            deflate,
        }
    }
}

/// A dataset to add: value, layout and fill value.
#[derive(Clone, Debug, PartialEq)]
pub struct NewDataset {
    /// Elements and shape.
    pub value: Value,
    /// Storage layout.
    pub layout: Layout,
    /// The fill value property (one element of the dataset's type); `None`
    /// is the library default (zeros).
    pub fill_value: Option<Data>,
}

impl NewDataset {
    /// A dataset with the default fill value.
    pub fn new(value: Value, layout: Layout) -> Self {
        Self {
            value,
            layout,
            fill_value: None,
        }
    }

    /// The same dataset with a fill value property.
    pub fn with_fill_value(mut self, fill_value: Data) -> Self {
        self.fill_value = Some(fill_value);
        self
    }
}

struct Attr {
    name: String,
    value: Value,
}

enum Kind {
    Group { links: Vec<(String, ObjectId)> },
    Dataset(NewDataset),
}

struct Node {
    kind: Kind,
    attrs: Vec<Attr>,
}

/// Builds an HDF5 file in memory. See the [module documentation](self).
pub struct Writer {
    nodes: Vec<Node>,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

/// A global heap object: its bytes, or object references resolved at
/// layout time.
enum HeapPayload {
    Bytes(Vec<u8>),
    Refs(Vec<ObjectId>),
}

impl HeapPayload {
    fn len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::Refs(refs) => refs.len() * 8,
        }
    }
}

/// Most objects per global heap collection (object indices are 16-bit, 0 is
/// free space).
const HEAP_MAX_OBJECTS: usize = 65_535;
/// A collection is closed once its objects exceed this many bytes.
const HEAP_TARGET_BYTES: usize = 1 << 20;
/// Smallest collection the HDF5 library reads (`H5HG_MINSIZE`).
const HEAP_MIN_BYTES: usize = 4096;

#[derive(Default)]
struct Heap {
    collections: Vec<Vec<HeapPayload>>,
    current_bytes: usize,
}

/// Where one variable-length element's data lives: collection and index.
#[derive(Clone, Copy)]
struct HeapSlot {
    collection: usize,
    index: u16,
}

impl Heap {
    fn push(&mut self, payload: HeapPayload) -> HeapSlot {
        let needs_new = match self.collections.last() {
            None => true,
            Some(last) => last.len() >= HEAP_MAX_OBJECTS || self.current_bytes >= HEAP_TARGET_BYTES,
        };
        if needs_new {
            self.collections.push(Vec::new());
            self.current_bytes = 0;
        }
        let collection = self.collections.len() - 1;
        self.current_bytes += 16 + payload.len().div_ceil(8) * 8;
        let objects = &mut self.collections[collection];
        objects.push(payload);
        HeapSlot {
            collection,
            index: objects.len() as u16,
        }
    }

    fn collection_size(objects: &[HeapPayload]) -> usize {
        let used: usize = 16
            + objects
                .iter()
                .map(|object| 16 + object.len().div_ceil(8) * 8)
                .sum::<usize>();
        used.max(HEAP_MIN_BYTES)
    }

    fn encode(objects: &[HeapPayload], resolve: &Resolver<'_>) -> Vec<u8> {
        let size = Self::collection_size(objects);
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(b"GCOL");
        out.extend_from_slice(&[1, 0, 0, 0]);
        out.extend_from_slice(&(size as u64).to_le_bytes());
        for (index, object) in objects.iter().enumerate() {
            out.extend_from_slice(&((index + 1) as u16).to_le_bytes());
            out.extend_from_slice(&1u16.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&(object.len() as u64).to_le_bytes());
            match object {
                HeapPayload::Bytes(bytes) => out.extend_from_slice(bytes),
                HeapPayload::Refs(refs) => {
                    for target in refs {
                        out.extend_from_slice(&resolve.object(*target).to_le_bytes());
                    }
                }
            }
            out.resize(out.len().div_ceil(8) * 8, 0);
        }
        let free = size - out.len();
        if free >= 16 {
            // Object 0: the free space, its size including this header.
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&(free as u64).to_le_bytes());
        }
        out.resize(size, 0);
        out
    }
}

/// Addresses known after layout (all zero while sizing).
struct Resolver<'a> {
    objects: &'a [u64],
    heaps: &'a [u64],
}

impl Resolver<'_> {
    fn object(&self, id: ObjectId) -> u64 {
        self.objects.get(id.0).copied().unwrap_or(0)
    }

    fn heap_id(&self, slot: HeapSlot, out: &mut Vec<u8>) {
        let address = self.heaps.get(slot.collection).copied().unwrap_or(0);
        out.extend_from_slice(&address.to_le_bytes());
        out.extend_from_slice(&u32::from(slot.index).to_le_bytes());
    }
}

/// Encoded elements of `data`; `slots` holds the heap slot of every
/// variable-length element, in order.
fn element_bytes(data: &Data, slots: &[HeapSlot], resolve: &Resolver<'_>) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * data.element_size());
    macro_rules! numbers {
        ($values:expr) => {
            for value in $values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        };
    }
    match data {
        Data::I8(v) => numbers!(v),
        Data::U8(v) => out.extend_from_slice(v),
        Data::I16(v) => numbers!(v),
        Data::U16(v) => numbers!(v),
        Data::I32(v) => numbers!(v),
        Data::U32(v) => numbers!(v),
        Data::I64(v) => numbers!(v),
        Data::U64(v) => numbers!(v),
        Data::F32(v) => numbers!(v),
        Data::F64(v) => numbers!(v),
        Data::FixedStrings { bytes, .. } => out.extend_from_slice(bytes),
        Data::VarStrings { values, .. } => {
            for (value, slot) in values.iter().zip(slots) {
                out.extend_from_slice(&(value.len() as u32).to_le_bytes());
                resolve.heap_id(*slot, &mut out);
            }
        }
        Data::ObjectRefs(targets) => {
            for target in targets {
                out.extend_from_slice(&resolve.object(*target).to_le_bytes());
            }
        }
        Data::ObjectRefLists(lists) => {
            for (list, slot) in lists.iter().zip(slots) {
                out.extend_from_slice(&(list.len() as u32).to_le_bytes());
                resolve.heap_id(*slot, &mut out);
            }
        }
        Data::DimensionScaleRefs(pairs) => {
            for (target, dimension) in pairs {
                out.extend_from_slice(&resolve.object(*target).to_le_bytes());
                out.extend_from_slice(&dimension.to_le_bytes());
                out.extend_from_slice(&[0; 4]);
            }
        }
        Data::Bools(values) => out.extend(values.iter().map(|value| u8::from(*value))),
        Data::Compound(members) => {
            let columns: Vec<(Vec<u8>, usize)> = members
                .iter()
                .map(|(_, column)| (element_bytes(column, &[], resolve), column.element_size()))
                .collect();
            for row in 0..data.len() {
                for (bytes, size) in &columns {
                    out.extend_from_slice(&bytes[row * size..(row + 1) * size]);
                }
            }
        }
    }
    out
}

/// Heap objects of a value's variable-length elements.
fn heap_slots(data: &Data, heap: &mut Heap) -> Vec<HeapSlot> {
    match data {
        Data::VarStrings { values, .. } => values
            .iter()
            .map(|value| heap.push(HeapPayload::Bytes(value.as_bytes().to_vec())))
            .collect(),
        Data::ObjectRefLists(lists) => lists
            .iter()
            .map(|list| heap.push(HeapPayload::Refs(list.clone())))
            .collect(),
        _ => Vec::new(),
    }
}

fn check_value(value: &Value, what: &str) -> Result<(), WriteError> {
    if let Data::Compound(members) = &value.data {
        if members.is_empty() || members.len() > usize::from(u16::MAX) {
            return Err(WriteError::Invalid(format!(
                "{what}: a compound needs 1 to 65,535 members"
            )));
        }
        let rows = members[0].1.len();
        for (name, column) in members {
            if name.is_empty() || name.contains('\0') {
                return Err(WriteError::Invalid(format!(
                    "{what}: invalid compound member name {name:?}"
                )));
            }
            if column.len() != rows || column.is_addressed() || matches!(column, Data::Compound(_))
            {
                return Err(WriteError::Invalid(format!(
                    "{what}: compound member {name} must be a fixed-size column of {rows} values"
                )));
            }
            check_value(&Value::vector(column.clone()), what)?;
        }
    }
    let expected = value
        .shape
        .element_count()
        .ok_or_else(|| WriteError::TooLarge(format!("{what}: element count overflows")))?;
    if let Data::FixedStrings { bytes, size, .. } = &value.data {
        if *size == 0 {
            return Err(WriteError::Invalid(format!(
                "{what}: fixed-length strings need at least one byte"
            )));
        }
        if bytes.len() % size != 0 {
            return Err(WriteError::Invalid(format!(
                "{what}: {} string bytes are not a multiple of the string size {size}",
                bytes.len()
            )));
        }
        if u32::try_from(*size).is_err() {
            return Err(WriteError::TooLarge(format!("{what}: string size {size}")));
        }
    }
    if value.data.len() != expected {
        return Err(WriteError::Invalid(format!(
            "{what}: {} elements for a shape of {expected}",
            value.data.len()
        )));
    }
    if let Shape::Simple(dims) = &value.shape
        && dims.len() > 32
    {
        return Err(WriteError::TooLarge(format!("{what}: rank {}", dims.len())));
    }
    Ok(())
}

/// The dataset storage decided before layout.
enum Storage {
    Contiguous {
        size: usize,
    },
    Compact,
    Chunked {
        chunk: Vec<u64>,
        chunks: Vec<chunked::StoredChunk>,
        nodes: Vec<chunked::Node>,
        shuffle: bool,
        deflate: Option<u32>,
    },
    Unallocated {
        size: usize,
    },
}

/// Per-object layout facts.
struct Planned {
    /// Heap slots of each attribute's value.
    attr_slots: Vec<Vec<HeapSlot>>,
    /// Heap slots of the dataset's value.
    data_slots: Vec<HeapSlot>,
    storage: Option<Storage>,
    header_size: usize,
}

/// Addresses assigned to one dataset's storage.
#[derive(Default, Clone)]
struct StorageAddresses {
    data: u64,
    nodes: Vec<u64>,
    chunks: Vec<u64>,
}

impl Writer {
    /// A writer holding only the root group.
    pub fn new() -> Self {
        Self {
            nodes: vec![Node {
                kind: Kind::Group { links: Vec::new() },
                attrs: Vec::new(),
            }],
        }
    }

    /// The root group.
    pub fn root(&self) -> ObjectId {
        ObjectId(0)
    }

    fn check_link(&self, parent: ObjectId, name: &str) -> Result<(), WriteError> {
        if name.is_empty() || name == "." || name.contains('/') || name.contains('\0') {
            return Err(WriteError::InvalidName(name.to_owned()));
        }
        match self.nodes.get(parent.0) {
            Some(Node {
                kind: Kind::Group { links },
                ..
            }) => {
                if links.iter().any(|(existing, _)| existing == name) {
                    return Err(WriteError::Duplicate(name.to_owned()));
                }
                if links.len() >= usize::from(u16::MAX) {
                    return Err(WriteError::TooLarge(format!(
                        "more than {} links in one group",
                        u16::MAX
                    )));
                }
                Ok(())
            }
            Some(_) => Err(WriteError::NotAGroup(parent.0)),
            None => Err(WriteError::UnknownObject(parent.0)),
        }
    }

    fn link(&mut self, parent: ObjectId, name: &str, node: Node) -> ObjectId {
        let id = ObjectId(self.nodes.len());
        self.nodes.push(node);
        if let Some(Node {
            kind: Kind::Group { links },
            ..
        }) = self.nodes.get_mut(parent.0)
        {
            links.push((name.to_owned(), id));
        }
        id
    }

    /// Add a group `name` below `parent`.
    pub fn add_group(&mut self, parent: ObjectId, name: &str) -> Result<ObjectId, WriteError> {
        self.check_link(parent, name)?;
        Ok(self.link(
            parent,
            name,
            Node {
                kind: Kind::Group { links: Vec::new() },
                attrs: Vec::new(),
            },
        ))
    }

    /// Add a dataset `name` below `parent`.
    pub fn add_dataset(
        &mut self,
        parent: ObjectId,
        name: &str,
        dataset: NewDataset,
    ) -> Result<ObjectId, WriteError> {
        self.check_link(parent, name)?;
        let what = format!("dataset {name}");
        match &dataset.layout {
            Layout::Unallocated => {
                if !dataset.value.data.is_empty() {
                    return Err(WriteError::Invalid(format!(
                        "{what}: an unallocated dataset takes no elements"
                    )));
                }
                if dataset.value.shape.element_count().is_none() {
                    return Err(WriteError::TooLarge(format!("{what}: element count")));
                }
            }
            layout => {
                check_value(&dataset.value, &what)?;
                if let Layout::Chunked { chunk, deflate, .. } = layout {
                    let Shape::Simple(dims) = &dataset.value.shape else {
                        return Err(WriteError::Invalid(format!(
                            "{what}: only simple dataspaces can be chunked"
                        )));
                    };
                    if dims.is_empty()
                        || chunk.len() != dims.len()
                        || chunk.iter().any(|c| *c == 0 || u32::try_from(*c).is_err())
                    {
                        return Err(WriteError::Invalid(format!(
                            "{what}: chunk {chunk:?} does not fit dimensions {dims:?}"
                        )));
                    }
                    if dataset.value.data.is_addressed() {
                        return Err(WriteError::Unsupported(format!(
                            "{what}: chunked variable-length or reference data"
                        )));
                    }
                    if deflate.is_some_and(|level| level > 9) {
                        return Err(WriteError::Invalid(format!(
                            "{what}: deflate level above 9"
                        )));
                    }
                }
            }
        }
        if let Some(fill) = &dataset.fill_value
            && (fill.len() != 1
                || std::mem::discriminant(fill) != std::mem::discriminant(&dataset.value.data)
                || fill.element_size() != dataset.value.data.element_size()
                || fill.is_addressed())
        {
            return Err(WriteError::Invalid(format!(
                "{what}: the fill value must be one element of the dataset's type"
            )));
        }
        Ok(self.link(
            parent,
            name,
            Node {
                kind: Kind::Dataset(dataset),
                attrs: Vec::new(),
            },
        ))
    }

    /// Add attribute `name` to a group or dataset.
    pub fn add_attribute(
        &mut self,
        object: ObjectId,
        name: &str,
        value: Value,
    ) -> Result<(), WriteError> {
        if name.is_empty() || name.contains('\0') {
            return Err(WriteError::InvalidAttributeName(name.to_owned()));
        }
        check_value(&value, &format!("attribute {name}"))?;
        let node = self
            .nodes
            .get_mut(object.0)
            .ok_or(WriteError::UnknownObject(object.0))?;
        if node.attrs.iter().any(|attr| attr.name == name) {
            return Err(WriteError::Duplicate(name.to_owned()));
        }
        if node.attrs.len() >= usize::from(u16::MAX) {
            return Err(WriteError::TooLarge(format!(
                "more than {} attributes on one object",
                u16::MAX
            )));
        }
        node.attrs.push(Attr {
            name: name.to_owned(),
            value,
        });
        Ok(())
    }

    /// `true` when `object` already has an attribute `name`.
    pub fn has_attribute(&self, object: ObjectId, name: &str) -> bool {
        self.nodes
            .get(object.0)
            .is_some_and(|node| node.attrs.iter().any(|attr| attr.name == name))
    }

    /// Lay the file out and return its bytes.
    pub fn finish(self) -> Result<Vec<u8>, WriteError> {
        // References must point at objects of this writer.
        for node in &self.nodes {
            let datasets = match &node.kind {
                Kind::Dataset(dataset) => Some(&dataset.value.data),
                Kind::Group { .. } => None,
            };
            for data in node
                .attrs
                .iter()
                .map(|attr| &attr.value.data)
                .chain(datasets)
            {
                let ids: Vec<ObjectId> = match data {
                    Data::ObjectRefs(ids) => ids.clone(),
                    Data::ObjectRefLists(lists) => lists.iter().flatten().copied().collect(),
                    Data::DimensionScaleRefs(pairs) => pairs.iter().map(|(id, _)| *id).collect(),
                    _ => Vec::new(),
                };
                if let Some(bad) = ids.iter().find(|id| id.0 >= self.nodes.len()) {
                    return Err(WriteError::UnknownObject(bad.0));
                }
            }
        }

        // Pass 1: heap objects, storage and header sizes.
        let mut heap = Heap::default();
        let zero = Resolver {
            objects: &[],
            heaps: &[],
        };
        let mut planned = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let attr_slots: Vec<Vec<HeapSlot>> = node
                .attrs
                .iter()
                .map(|attr| heap_slots(&attr.value.data, &mut heap))
                .collect();
            let (data_slots, storage) = match &node.kind {
                Kind::Group { .. } => (Vec::new(), None),
                Kind::Dataset(dataset) => {
                    let slots = if matches!(dataset.layout, Layout::Unallocated) {
                        Vec::new()
                    } else {
                        heap_slots(&dataset.value.data, &mut heap)
                    };
                    (slots, Some(plan_storage(dataset)?))
                }
            };
            let mut plan = Planned {
                attr_slots,
                data_slots,
                storage,
                header_size: 0,
            };
            let messages = self.messages(node, &plan, &zero, &StorageAddresses::default())?;
            plan.header_size = encode::object_header_size(messages.iter().map(|m| m.body.len()));
            planned.push(plan);
        }

        // Layout: superblock, object headers, heap collections, then each
        // dataset's B-tree nodes and chunks or contiguous block.
        let mut next = SUPERBLOCK_SIZE as u64;
        let mut object_addresses = Vec::with_capacity(planned.len());
        for plan in &planned {
            object_addresses.push(next);
            next += plan.header_size as u64;
        }
        let mut heap_addresses = Vec::with_capacity(heap.collections.len());
        for objects in &heap.collections {
            heap_addresses.push(next);
            next += Heap::collection_size(objects) as u64;
        }
        let mut storage_addresses = vec![StorageAddresses::default(); planned.len()];
        for (plan, addresses) in planned.iter().zip(&mut storage_addresses) {
            match &plan.storage {
                Some(Storage::Contiguous { size }) => {
                    if *size == 0 {
                        addresses.data = UNDEF;
                    } else {
                        addresses.data = next;
                        next += *size as u64;
                    }
                }
                Some(Storage::Chunked {
                    chunk,
                    chunks,
                    nodes,
                    ..
                }) => {
                    let node_size = chunked::node_size(chunk.len()) as u64;
                    for _ in nodes {
                        addresses.nodes.push(next);
                        next += node_size;
                    }
                    for stored in chunks {
                        addresses.chunks.push(next);
                        next += stored.bytes.len() as u64;
                    }
                }
                Some(Storage::Unallocated { .. }) => addresses.data = UNDEF,
                Some(Storage::Compact) | None => {}
            }
        }
        let end = next;
        let total = usize::try_from(end)
            .map_err(|_| WriteError::TooLarge("file larger than addressable memory".to_owned()))?;

        // Pass 2: encode with the real addresses.
        let resolve = Resolver {
            objects: &object_addresses,
            heaps: &heap_addresses,
        };
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&encode::superblock(end, object_addresses[0]));
        for ((node, plan), addresses) in self.nodes.iter().zip(&planned).zip(&storage_addresses) {
            let messages = self.messages(node, plan, &resolve, addresses)?;
            let header = encode::object_header(&messages);
            if header.len() != plan.header_size {
                return Err(WriteError::Invalid(
                    "object header size changed between passes".to_owned(),
                ));
            }
            out.extend_from_slice(&header);
        }
        for objects in &heap.collections {
            out.extend_from_slice(&Heap::encode(objects, &resolve));
        }
        for ((node, plan), addresses) in self.nodes.iter().zip(&planned).zip(&storage_addresses) {
            let Kind::Dataset(dataset) = &node.kind else {
                continue;
            };
            match &plan.storage {
                Some(Storage::Contiguous { size }) if *size > 0 => {
                    let bytes = element_bytes(&dataset.value.data, &plan.data_slots, &resolve);
                    out.extend_from_slice(&bytes);
                }
                Some(Storage::Chunked {
                    chunk,
                    chunks,
                    nodes,
                    ..
                }) => {
                    let element_size = dataset.value.data.element_size() as u64;
                    for index in 0..nodes.len() {
                        out.extend_from_slice(&chunked::encode_node(
                            nodes,
                            index,
                            &addresses.nodes,
                            chunks,
                            &addresses.chunks,
                            chunk,
                            element_size,
                        ));
                    }
                    for stored in chunks {
                        out.extend_from_slice(&stored.bytes);
                    }
                }
                _ => {}
            }
        }
        if out.len() != total {
            return Err(WriteError::Invalid(format!(
                "laid out {total} bytes but wrote {}",
                out.len()
            )));
        }
        Ok(out)
    }

    /// The header messages of one object.
    fn messages(
        &self,
        node: &Node,
        plan: &Planned,
        resolve: &Resolver<'_>,
        addresses: &StorageAddresses,
    ) -> Result<Vec<Message>, WriteError> {
        let mut messages = Vec::new();
        match &node.kind {
            Kind::Group { links } => {
                messages.push(Message::new(
                    encode::MSG_LINK_INFO,
                    encode::link_info(links.len() as u64),
                ));
                messages.push(Message::new(
                    encode::MSG_GROUP_INFO,
                    encode::group_info(links.len()),
                ));
                for (order, (name, target)) in links.iter().enumerate() {
                    messages.push(Message::new(
                        encode::MSG_LINK,
                        encode::hard_link(name, order as u64, resolve.object(*target)),
                    ));
                }
            }
            Kind::Dataset(dataset) => {
                let value = &dataset.value;
                messages.push(Message::new(
                    encode::MSG_DATASPACE,
                    encode::dataspace(&value.shape),
                ));
                messages.push(Message::constant(
                    encode::MSG_DATATYPE,
                    encode::datatype(&value.data),
                ));
                let fill = dataset
                    .fill_value
                    .as_ref()
                    .map(|fill| element_bytes(fill, &[], resolve));
                let storage = plan.storage.as_ref().ok_or_else(|| {
                    WriteError::Invalid("dataset without planned storage".to_owned())
                })?;
                let alloc = match storage {
                    Storage::Compact => AllocTime::Early,
                    Storage::Chunked { .. } => AllocTime::Incremental,
                    Storage::Contiguous { .. } | Storage::Unallocated { .. } => AllocTime::Late,
                };
                messages.push(Message::constant(
                    encode::MSG_FILL,
                    encode::fill_value(alloc, fill.as_deref()),
                ));
                let element_size = value.data.element_size();
                let layout = match storage {
                    Storage::Contiguous { size } | Storage::Unallocated { size } => {
                        encode::contiguous_layout(addresses.data, *size as u64)
                    }
                    Storage::Compact => {
                        let raw = element_bytes(&value.data, &plan.data_slots, resolve);
                        encode::compact_layout(&raw)
                    }
                    Storage::Chunked { chunk, .. } => {
                        let root = addresses.nodes.last().copied().unwrap_or(UNDEF);
                        encode::chunked_layout(root, chunk, element_size as u32)
                    }
                };
                messages.push(Message::new(encode::MSG_LAYOUT, layout));
                if let Storage::Chunked {
                    shuffle, deflate, ..
                } = storage
                    && (*shuffle || deflate.is_some())
                {
                    let shuffle = (*shuffle && element_size > 1).then_some(element_size as u32);
                    messages.push(Message::constant(
                        encode::MSG_FILTERS,
                        encode::filter_pipeline(shuffle, *deflate),
                    ));
                }
            }
        }
        messages.push(Message::new(
            encode::MSG_ATTRIBUTE_INFO,
            encode::attribute_info(node.attrs.len() as u16),
        ));
        for (order, (attr, slots)) in node.attrs.iter().zip(&plan.attr_slots).enumerate() {
            let datatype = encode::datatype(&attr.value.data);
            let dataspace = encode::dataspace(&attr.value.shape);
            let data = element_bytes(&attr.value.data, slots, resolve);
            let body = encode::attribute(
                &attr.name,
                &datatype,
                &dataspace,
                !attr.name.is_ascii(),
                &data,
            );
            if body.len() > usize::from(u16::MAX) {
                return Err(WriteError::TooLarge(format!(
                    "attribute {} needs {} bytes; compact attributes hold at most 65,535",
                    attr.name,
                    body.len()
                )));
            }
            let mut message = Message::new(encode::MSG_ATTRIBUTE, body);
            message.creation_order = order as u16;
            messages.push(message);
        }
        for message in &messages {
            if message.body.len() > usize::from(u16::MAX) {
                return Err(WriteError::TooLarge(format!(
                    "header message of type {:#04x} needs {} bytes",
                    message.kind,
                    message.body.len()
                )));
            }
        }
        Ok(messages)
    }
}

/// Decide a dataset's storage; chunked data is split and filtered here.
fn plan_storage(dataset: &NewDataset) -> Result<Storage, WriteError> {
    let value = &dataset.value;
    let element_size = value.data.element_size();
    let count = value
        .shape
        .element_count()
        .ok_or_else(|| WriteError::TooLarge("dataset element count".to_owned()))?;
    let size = count
        .checked_mul(element_size)
        .ok_or_else(|| WriteError::TooLarge("dataset byte size".to_owned()))?;
    Ok(match &dataset.layout {
        Layout::Contiguous => Storage::Contiguous { size },
        Layout::Unallocated => Storage::Unallocated { size },
        Layout::Compact => {
            if size > 60_000 {
                return Err(WriteError::TooLarge(format!(
                    "compact dataset of {size} bytes (at most 60,000)"
                )));
            }
            Storage::Compact
        }
        Layout::Chunked {
            chunk,
            shuffle,
            deflate,
        } => {
            let Shape::Simple(dims) = &value.shape else {
                return Err(WriteError::Invalid("chunked scalar dataset".to_owned()));
            };
            let raw = element_bytes(
                &value.data,
                &[],
                &Resolver {
                    objects: &[],
                    heaps: &[],
                },
            );
            let fill = match &dataset.fill_value {
                Some(fill) => element_bytes(
                    fill,
                    &[],
                    &Resolver {
                        objects: &[],
                        heaps: &[],
                    },
                ),
                None => vec![0; element_size],
            };
            let chunks =
                chunked::split(&raw, dims, chunk, element_size, &fill, *shuffle, *deflate)?;
            let nodes = chunked::btree(chunks.len());
            Storage::Chunked {
                chunk: chunk.clone(),
                chunks,
                nodes,
                shuffle: *shuffle,
                deflate: *deflate,
            }
        }
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
