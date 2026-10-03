//! The file view: object graph, cached object headers, attributes and
//! dataset reads.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::btree1;
use crate::btree2;
use crate::bytes::{Cursor, UNDEFINED_ADDR};
use crate::chunks::{self, Geometry};
use crate::dataspace::{self, Dataspace};
use crate::datatype::{self, Datatype};
use crate::error::{Error, Result, invalid, limit, unsupported};
use crate::filters::{self, Filter};
use crate::fractal::FractalHeap;
use crate::header::{
    self, FLAG_SHARED, MSG_ATTRIBUTE, MSG_ATTRIBUTE_INFO, MSG_DATASPACE, MSG_DATATYPE,
    MSG_EXTERNAL_FILES, MSG_FILL, MSG_FILL_OLD, MSG_FILTERS, MSG_GROUP_INFO, MSG_LAYOUT, MSG_LINK,
    MSG_LINK_INFO, MSG_SYMBOL_TABLE, ObjectHeader,
};
use crate::heap::{GlobalHeaps, LocalHeap};
use crate::layout::{self, Layout, StorageLayout};
use crate::limits::{
    MAX_ATTRIBUTE_BYTES, MAX_ATTRIBUTES_PER_OBJECT, MAX_DATASET_BYTES, MAX_FILE_ATTRIBUTE_BYTES,
    MAX_FILE_ENTRIES, MAX_FILE_NAME_BYTES, MAX_GROUP_DEPTH, MAX_GROUP_ENTRIES, MAX_OBJECTS,
    MAX_PATH_BYTES,
};
use crate::link::{self, Link, LinkTarget};
use crate::space::Space;
use crate::superblock;
use crate::values::{Decoder, Values};

/// What an object header describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ObjectKind {
    /// A group.
    Group,
    /// A dataset.
    Dataset,
    /// A committed (named) datatype.
    Datatype,
    /// An object header of none of the above.
    Other,
}

/// A decoded attribute.
#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
    name: String,
    datatype: Datatype,
    dims: Vec<usize>,
    null: bool,
    values: Values,
    creation_order: Option<u64>,
}

impl Attribute {
    /// Attribute name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Stored datatype (a committed datatype is resolved).
    pub fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    /// Dataspace dimensions; empty for a scalar (and a null dataspace).
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// True for a null dataspace (an attribute with no value).
    pub fn is_null(&self) -> bool {
        self.null
    }

    /// The values, in the stored type.
    pub fn values(&self) -> &Values {
        &self.values
    }

    /// Creation order, when the object tracks it.
    pub fn creation_order(&self) -> Option<u64> {
        self.creation_order
    }

    /// The first string of a string attribute.
    pub fn as_str(&self) -> Option<String> {
        self.values.strings()?.into_iter().next()
    }

    /// A one-element numeric attribute widened to `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        (self.values.len() == 1)
            .then(|| self.values.get_f64(0))
            .flatten()
    }

    /// A one-element integer attribute, or a one-element float with no
    /// fractional part, as `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        if self.values.len() != 1 {
            return None;
        }
        self.values.get_i64(0).or_else(|| {
            let value = self.values.get_f64(0)?;
            (value.fract() == 0.0 && value.abs() < 9.2e18).then_some(value as i64)
        })
    }
}

/// Metadata of one object: kind, attributes and (for groups) links.
#[derive(Clone, Debug)]
pub struct Object {
    address: u64,
    kind: ObjectKind,
    header_version: u8,
    attributes: Vec<Attribute>,
    links: Vec<Link>,
    path: String,
}

impl Object {
    /// Object header address (relative to the superblock).
    pub fn address(&self) -> u64 {
        self.address
    }

    /// What the object is.
    pub fn kind(&self) -> ObjectKind {
        self.kind
    }

    /// Object header version (1 or 2).
    pub fn header_version(&self) -> u8 {
        self.header_version
    }

    /// Attributes in the order HDF5 iterates them by creation index (what
    /// netCDF-C lists): creation order when the object tracks it; otherwise
    /// header order (compact storage) or the name index's order (dense
    /// storage, by name hash). [`Attribute::creation_order`] is set only
    /// when the object tracks it.
    pub fn attributes(&self) -> &[Attribute] {
        &self.attributes
    }

    /// The attribute called `name`.
    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name == name)
    }

    /// Links of a group: creation order when tracked, otherwise name order.
    pub fn links(&self) -> &[Link] {
        &self.links
    }

    /// The first path under which the walk from the root reached the
    /// object.
    pub fn path(&self) -> &str {
        &self.path
    }
}

/// Dataset metadata, without the data.
#[derive(Clone, Debug)]
pub struct DatasetInfo {
    /// Current dimensions (empty for a scalar).
    pub dims: Vec<usize>,
    /// Maximum dimensions ([`crate::UNLIMITED`] = unlimited), when stored.
    pub max_dims: Option<Vec<u64>>,
    /// Element datatype (a committed datatype is resolved).
    pub datatype: Datatype,
    /// Address of the committed datatype object, when the dataset uses one.
    pub committed_datatype: Option<u64>,
    /// Storage layout.
    pub layout: StorageLayout,
    /// Filter pipeline (empty when unfiltered).
    pub filters: Vec<Filter>,
    /// Fill value bytes (one element), when defined.
    pub fill_value: Option<Vec<u8>>,
    /// True for a null dataspace.
    pub null: bool,
}

/// One stored chunk of a chunked dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkLocation {
    /// Element offset of the chunk's first element, per dimension.
    pub offsets: Vec<u64>,
    /// Filters skipped for this chunk (bit N = filter N).
    pub filter_mask: u32,
    /// Absolute file offset of the stored chunk.
    pub file_offset: u64,
    /// Stored (filtered) size in bytes.
    pub stored_size: usize,
}

/// A dataset read in full.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// Dimensions, row-major (empty for a scalar).
    pub dims: Vec<usize>,
    /// Element datatype.
    pub datatype: Datatype,
    /// The elements.
    pub values: Values,
}

/// How [`H5File::open_with`] reads a file.
///
/// The default verifies every checksum, as the HDF5 library does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct OpenOptions {
    /// Verify the Jenkins lookup3 checksums of the metadata structures that
    /// carry one (superblock versions 2 and 3, version 2 object headers and
    /// their continuation blocks, v2 B-tree headers and nodes, fractal heap
    /// headers and blocks, fixed and extensible array blocks). Default
    /// `true`. Turning it off reads a file whose metadata bytes were altered
    /// after writing as far as the structures still parse: every size and
    /// limit is checked either way, so a file read without checksums is
    /// still bounded. Fletcher-32 data checksums are always verified. The
    /// fuzz harness turns it off so mutated inputs reach the parsers behind
    /// the checksums.
    pub verify_metadata_checksums: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            verify_metadata_checksums: true,
        }
    }
}

impl OpenOptions {
    /// Set [`OpenOptions::verify_metadata_checksums`].
    #[must_use]
    pub fn with_metadata_checksums(mut self, verify: bool) -> Self {
        self.verify_metadata_checksums = verify;
        self
    }
}

/// Read-only HDF5 file view over a byte slice.
///
/// [`H5File::open`] walks every group from the root once, parses each
/// object header once, resolves each committed datatype once and decodes
/// every attribute once; lookups afterwards never re-parse the file.
/// Dataset data is read on demand.
pub struct H5File<'a> {
    space: Space<'a>,
    superblock_version: u8,
    objects: Vec<Object>,
    headers: Vec<ObjectHeader<'a>>,
    by_address: HashMap<u64, usize>,
    paths: BTreeMap<String, usize>,
    /// Committed datatypes by object header address, each parsed once.
    committed: CommittedTypes,
}

/// Running totals of one [`H5File::open`], checked against the per-file
/// limits.
struct Walk<'a> {
    heaps: GlobalHeaps<'a>,
    attribute_bytes: usize,
    path_bytes: usize,
    /// Links and attributes decoded ([`MAX_FILE_ENTRIES`]).
    entries: usize,
    /// Bytes of link and attribute names decoded ([`MAX_FILE_NAME_BYTES`]).
    name_bytes: usize,
    /// Fractal heaps by header address, each parsed once per file.
    fractal: HashMap<u64, FractalHeap>,
    /// Fractal heap work ([`crate::fractal::charge`]).
    heap_work: usize,
}

impl Walk<'_> {
    /// Count one decoded link or attribute and its name.
    fn entry(&mut self, name_len: usize) -> Result<()> {
        self.entries += 1;
        self.name_bytes = self.name_bytes.saturating_add(name_len);
        if self.entries > MAX_FILE_ENTRIES {
            return Err(limit(format!(
                "HDF5 file has more than {MAX_FILE_ENTRIES} links and attributes (limit)"
            )));
        }
        if self.name_bytes > MAX_FILE_NAME_BYTES {
            return Err(limit(format!(
                "HDF5 link and attribute names exceed {MAX_FILE_NAME_BYTES} bytes per file (limit)"
            )));
        }
        Ok(())
    }

    /// The fractal heap at `address`, parsed on first use.
    fn fractal_heap(&mut self, space: &Space<'_>, address: u64) -> Result<&FractalHeap> {
        match self.fractal.entry(address) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => Ok(entry.insert(FractalHeap::parse(space, address)?)),
        }
    }
}

/// Committed datatypes by object header address, each parsed once.
type CommittedTypes = HashMap<u64, Datatype>;

impl<'a> H5File<'a> {
    /// Parse the superblock, walk the group tree from the root and decode
    /// every attribute, verifying every checksum
    /// ([`OpenOptions::default`]).
    pub fn open(bytes: &'a [u8]) -> Result<Self> {
        Self::open_with(bytes, OpenOptions::default())
    }

    /// [`H5File::open`] with options.
    pub fn open_with(bytes: &'a [u8], options: OpenOptions) -> Result<Self> {
        let verify_checksums = options.verify_metadata_checksums;
        let superblock = superblock::find(bytes, verify_checksums)?;
        let space = Space {
            bytes,
            base: superblock.base,
            offset_size: superblock.offset_size,
            length_size: superblock.length_size,
            verify_checksums,
        };
        let mut file = Self {
            space,
            superblock_version: superblock.version,
            objects: Vec::new(),
            headers: Vec::new(),
            by_address: HashMap::new(),
            paths: BTreeMap::new(),
            committed: HashMap::new(),
        };
        let mut walk = Walk {
            heaps: GlobalHeaps::default(),
            attribute_bytes: 0,
            path_bytes: 0,
            entries: 0,
            name_bytes: 0,
            fractal: HashMap::new(),
            heap_work: 0,
        };
        let root = file.load(superblock.root, "/", &mut walk)?;
        file.objects[root].kind = ObjectKind::Group;
        file.paths.insert("/".to_owned(), root);
        let mut visited = HashSet::from([root]);
        let mut soft = Vec::new();
        file.walk_group(root, "", 0, &mut visited, &mut soft, &mut walk)?;
        file.resolve_soft_links(soft);
        file.decode_attributes(&mut walk)?;
        Ok(file)
    }

    /// Superblock version (0-3).
    pub fn superblock_version(&self) -> u8 {
        self.superblock_version
    }

    /// Size of file addresses in bytes.
    pub fn offset_size(&self) -> usize {
        self.space.offset_size
    }

    /// Every indexed path and its object, in path order.
    pub fn objects(&self) -> impl Iterator<Item = (&str, &Object)> {
        self.paths
            .iter()
            .map(|(path, index)| (path.as_str(), &self.objects[*index]))
    }

    /// The object at an absolute path (`"/"`, `"/a/b"`).
    pub fn object(&self, path: &str) -> Option<&Object> {
        self.paths.get(path).map(|index| &self.objects[*index])
    }

    /// The object whose header is at `address` (for resolving object
    /// references).
    pub fn object_at(&self, address: u64) -> Option<&Object> {
        self.by_address
            .get(&address)
            .map(|index| &self.objects[*index])
    }

    /// True when a path names an object.
    pub fn has_object(&self, path: &str) -> bool {
        self.paths.contains_key(path)
    }

    /// Names of the links of the group at `path` (empty when not a group).
    pub fn child_names(&self, path: &str) -> Vec<String> {
        self.object(path)
            .map(|object| object.links.iter().map(|link| link.name.clone()).collect())
            .unwrap_or_default()
    }

    /// Attributes of the object at `path` (empty when it does not exist).
    pub fn attrs(&self, path: &str) -> &[Attribute] {
        self.object(path).map_or(&[], |object| &object.attributes)
    }

    /// One attribute of the object at `path`.
    pub fn attr(&self, path: &str, name: &str) -> Option<&Attribute> {
        self.object(path)?.attribute(name)
    }

    /// Dataset metadata.
    pub fn dataset_info(&self, path: &str) -> Result<DatasetInfo> {
        let index = *self
            .paths
            .get(path)
            .ok_or_else(|| Error::NotFound(path.to_owned()))?;
        self.info_of(index, path).map(|(info, _)| info)
    }

    /// Read a whole dataset.
    pub fn dataset(&self, path: &str) -> Result<Dataset> {
        let index = *self
            .paths
            .get(path)
            .ok_or_else(|| Error::NotFound(path.to_owned()))?;
        let (info, layout) = self.info_of(index, path)?;
        let count = if info.null {
            0
        } else {
            info.dims
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
                .ok_or_else(|| invalid(0, "HDF5 dataset element count overflow"))?
        };
        let element_size = info.datatype.size();
        let byte_len = checked_bytes(count, element_size, "HDF5 dataset")?;
        checked_bytes(count, decoded_width(&info.datatype), "decoded HDF5 dataset")?;
        let fill = |len: usize| -> Vec<u8> {
            match &info.fill_value {
                Some(value) if value.len() == element_size && value.iter().any(|b| *b != 0) => {
                    value.iter().copied().cycle().take(len).collect()
                }
                _ => vec![0u8; len],
            }
        };
        let raw: Vec<u8> = match layout {
            Layout::Compact(data) => data
                .get(..byte_len)
                .ok_or_else(|| {
                    invalid(
                        0,
                        format!(
                            "dataset '{path}' raw stream too short: {} < {byte_len}",
                            data.len()
                        ),
                    )
                })?
                .to_vec(),
            Layout::Contiguous { address, size } => {
                if address == UNDEFINED_ADDR {
                    fill(byte_len)
                } else {
                    let stored = size.map_or(byte_len, |size| (size as usize).min(byte_len));
                    let data = self.space.slice(address, stored)?;
                    if data.len() < byte_len {
                        return Err(invalid(
                            0,
                            format!(
                                "dataset '{path}' raw stream too short: {} < {byte_len}",
                                data.len()
                            ),
                        ));
                    }
                    data.to_vec()
                }
            }
            Layout::Chunked {
                chunk_dims,
                index,
                flags,
            } => {
                filters::check_supported(&info.filters)?;
                if chunk_dims.len() != info.dims.len() {
                    return Err(invalid(
                        0,
                        "HDF5 chunk dimensionality does not match dataset rank",
                    ));
                }
                let dims: Vec<u64> = info.dims.iter().map(|dim| *dim as u64).collect();
                let max_dims = info.max_dims.clone().unwrap_or_else(|| dims.clone());
                let geometry = Geometry {
                    dims: &dims,
                    max_dims: &max_dims,
                    chunk_dims: &chunk_dims,
                    element_size,
                };
                let mut out = fill(byte_len);
                if count > 0 {
                    let chunks = chunks::collect(&self.space, &index, &geometry)?;
                    chunks::assemble(
                        &self.space,
                        &chunks,
                        &geometry,
                        &info.filters,
                        flags & 0x01 != 0,
                        &mut out,
                    )?;
                }
                out
            }
            Layout::Virtual => return Err(unsupported("virtual dataset layout")),
        };
        let mut heaps = GlobalHeaps::default();
        let mut decoder = Decoder::new(self.space, &mut heaps);
        let values = decoder.decode(&info.datatype, &raw, count)?;
        Ok(Dataset {
            dims: info.dims,
            datatype: info.datatype,
            values,
        })
    }

    /// Every allocated chunk of a chunked dataset, in index order (empty
    /// for other layouts).
    pub fn chunk_locations(&self, path: &str) -> Result<Vec<ChunkLocation>> {
        let index = *self
            .paths
            .get(path)
            .ok_or_else(|| Error::NotFound(path.to_owned()))?;
        let (info, layout) = self.info_of(index, path)?;
        let Layout::Chunked {
            chunk_dims, index, ..
        } = layout
        else {
            return Ok(Vec::new());
        };
        let dims: Vec<u64> = info.dims.iter().map(|dim| *dim as u64).collect();
        let max_dims = info.max_dims.clone().unwrap_or_else(|| dims.clone());
        let geometry = Geometry {
            dims: &dims,
            max_dims: &max_dims,
            chunk_dims: &chunk_dims,
            element_size: info.datatype.size(),
        };
        chunks::collect(&self.space, &index, &geometry)?
            .into_iter()
            .map(|chunk| {
                // A chunk index may list chunks outside the extent: their
                // coordinates are the file's, and must not overflow.
                let offsets = chunk
                    .scaled
                    .iter()
                    .zip(&chunk_dims)
                    .map(|(scaled, dim)| {
                        scaled
                            .checked_mul(*dim as u64)
                            .ok_or_else(|| invalid(0, "HDF5 chunk offset overflows u64"))
                    })
                    .collect::<Result<Vec<u64>>>()?;
                Ok(ChunkLocation {
                    offsets,
                    filter_mask: chunk.filter_mask,
                    file_offset: self.space.abs(chunk.address)? as u64,
                    stored_size: chunk.size,
                })
            })
            .collect()
    }

    fn info_of(&self, index: usize, path: &str) -> Result<(DatasetInfo, Layout<'a>)> {
        let header = &self.headers[index];
        let space = &self.space;
        let dataspace = header
            .first(MSG_DATASPACE)
            .ok_or_else(|| invalid(0, format!("dataset '{path}' has no dataspace")))
            .and_then(|message| {
                dataspace::parse(message.body, message.offset, space.length_size)
            })?;
        let (datatype, committed_datatype) = match header.first(MSG_DATATYPE) {
            Some(message) if message.flags & FLAG_SHARED != 0 => {
                let address = shared_address(message.body, message.offset, space)?;
                let datatype = match self.committed.get(&address) {
                    Some(datatype) => datatype.clone(),
                    None => self.committed_datatype(address, message.offset)?,
                };
                (datatype, Some(address))
            }
            Some(message) => (
                datatype::parse(message.body, message.offset, space.offset_size)?,
                None,
            ),
            None => return Err(invalid(0, format!("dataset '{path}' has no datatype"))),
        };
        let layout = header
            .first(MSG_LAYOUT)
            .ok_or_else(|| invalid(0, format!("dataset '{path}' has no layout")))
            .and_then(|message| {
                layout::parse(
                    message.body,
                    message.offset,
                    space.offset_size,
                    space.length_size,
                )
            })?;
        if header.first(MSG_EXTERNAL_FILES).is_some() {
            return Err(unsupported(format!(
                "dataset '{path}' stores its data in external files"
            )));
        }
        let filters = match header.first(MSG_FILTERS) {
            Some(message) => filters::parse(message.body, message.offset)?,
            None => Vec::new(),
        };
        let fill_value = match (header.first(MSG_FILL), header.first(MSG_FILL_OLD)) {
            (Some(message), _) => parse_fill(message.body, message.offset)?,
            (None, Some(message)) => parse_fill_old(message.body, message.offset)?,
            (None, None) => None,
        };
        Ok((
            DatasetInfo {
                dims: dataspace.dims_usize(),
                max_dims: dataspace.max_dims.clone(),
                datatype,
                committed_datatype,
                layout: layout.describe(),
                filters,
                fill_value,
                null: dataspace.null,
            },
            layout,
        ))
    }

    /// A datatype message body, resolving a committed (shared) datatype
    /// through `cache` (each committed type is parsed once per file).
    fn message_datatype(
        &self,
        body: &[u8],
        offset: usize,
        flags: u8,
        cache: &mut CommittedTypes,
    ) -> Result<(Datatype, Option<u64>)> {
        if flags & FLAG_SHARED == 0 {
            return Ok((datatype::parse(body, offset, self.space.offset_size)?, None));
        }
        let address = shared_address(body, offset, &self.space)?;
        if let Some(datatype) = cache.get(&address) {
            return Ok((datatype.clone(), Some(address)));
        }
        let datatype = self.committed_datatype(address, offset)?;
        cache.insert(address, datatype.clone());
        Ok((datatype, Some(address)))
    }

    /// Parse the committed datatype whose object header is at `address`:
    /// from the header the group walk parsed, else from the file.
    fn committed_datatype(&self, address: u64, offset: usize) -> Result<Datatype> {
        #[cfg(test)]
        tests::COMMITTED_PARSES.with(|count| count.set(count.get() + 1));
        let parsed;
        let header = match self.by_address.get(&address) {
            Some(index) => &self.headers[*index],
            None => {
                parsed = header::parse(&self.space, address)?;
                &parsed
            }
        };
        let message = header
            .first(MSG_DATATYPE)
            .ok_or_else(|| invalid(offset, "committed datatype object has no datatype message"))?;
        if message.flags & FLAG_SHARED != 0 {
            return Err(invalid(
                offset,
                "committed datatype refers to another shared datatype",
            ));
        }
        datatype::parse(message.body, message.offset, self.space.offset_size)
    }

    // ----- object graph ------------------------------------------------

    /// Decode the attributes of every object, after the group walk.
    fn decode_attributes(&mut self, walk: &mut Walk<'a>) -> Result<()> {
        let mut cache = CommittedTypes::new();
        // The committed types of datasets (and of committed types that
        // refer to others), for `info_of` later.
        for header in &self.headers {
            if let Some(message) = header.first(MSG_DATATYPE)
                && message.flags & FLAG_SHARED != 0
            {
                self.message_datatype(message.body, message.offset, message.flags, &mut cache)?;
            }
        }
        for index in 0..self.objects.len() {
            let attributes = self
                .read_attributes(&self.headers[index], walk, &mut cache)
                .map_err(|err| match err {
                    Error::LimitExceeded(_) | Error::Unsupported(_) => err,
                    err => invalid(
                        self.space.abs(self.objects[index].address).unwrap_or(0),
                        format!(
                            "an attribute of HDF5 object '{}' cannot be read: {err}",
                            self.objects[index].path
                        ),
                    ),
                })?;
            self.objects[index].attributes = attributes;
        }
        self.committed = cache;
        Ok(())
    }

    /// Parse and cache the object at `address` (once per address).
    fn load(&mut self, address: u64, path: &str, walk: &mut Walk<'a>) -> Result<usize> {
        if let Some(index) = self.by_address.get(&address) {
            return Ok(*index);
        }
        if self.objects.len() >= MAX_OBJECTS {
            return Err(limit(format!(
                "HDF5 file indexes more than {MAX_OBJECTS} objects (limit)"
            )));
        }
        let header = header::parse(&self.space, address)?;
        let has = |kind| header.first(kind).is_some();
        let kind = if has(MSG_LAYOUT) {
            ObjectKind::Dataset
        } else if has(MSG_SYMBOL_TABLE)
            || has(MSG_LINK_INFO)
            || has(MSG_LINK)
            || has(MSG_GROUP_INFO)
        {
            ObjectKind::Group
        } else if has(MSG_DATATYPE) && !has(MSG_DATASPACE) {
            ObjectKind::Datatype
        } else {
            ObjectKind::Other
        };
        let links = if kind == ObjectKind::Group {
            self.read_links(&header, walk)?
        } else {
            Vec::new()
        };
        let index = self.objects.len();
        self.objects.push(Object {
            address,
            kind,
            header_version: header.version,
            attributes: Vec::new(),
            links,
            path: path.to_owned(),
        });
        self.headers.push(header);
        self.by_address.insert(address, index);
        Ok(index)
    }

    fn walk_group(
        &mut self,
        group: usize,
        prefix: &str,
        depth: usize,
        visited: &mut HashSet<usize>,
        soft: &mut Vec<(String, String)>,
        walk: &mut Walk<'a>,
    ) -> Result<()> {
        if depth > MAX_GROUP_DEPTH {
            return Err(limit(format!(
                "HDF5 group nesting is deeper than {MAX_GROUP_DEPTH} levels (limit)"
            )));
        }
        let links = self.objects[group].links.clone();
        for link in links {
            let path = format!("{prefix}/{}", link.name);
            if self.paths.contains_key(&path) {
                continue; // hard-link cycle guard
            }
            match link.target {
                LinkTarget::Hard(address) => {
                    walk.path_bytes = walk.path_bytes.saturating_add(path.len());
                    if walk.path_bytes > MAX_PATH_BYTES || self.paths.len() >= MAX_OBJECTS {
                        return Err(limit(format!(
                            "HDF5 file indexes more than {MAX_OBJECTS} objects or {MAX_PATH_BYTES} path bytes (limit)"
                        )));
                    }
                    let child = self.load(address, &path, walk)?;
                    self.paths.insert(path.clone(), child);
                    if self.objects[child].kind == ObjectKind::Group && visited.insert(child) {
                        self.walk_group(child, &path, depth + 1, visited, soft, walk)?;
                    }
                }
                LinkTarget::Soft(target) => {
                    let target = if target.starts_with('/') {
                        target
                    } else {
                        format!("{prefix}/{target}")
                    };
                    soft.push((path, target));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn resolve_soft_links(&mut self, soft: Vec<(String, String)>) {
        for (path, target) in soft {
            if let Some(index) = self.paths.get(&target).copied()
                && self.paths.len() < MAX_OBJECTS
            {
                self.paths.entry(path).or_insert(index);
            }
        }
    }

    /// Links of a group: old-style symbol table, compact link messages, or
    /// dense storage (fractal heap + v2 B-tree).
    fn read_links(&self, header: &ObjectHeader<'a>, walk: &mut Walk<'a>) -> Result<Vec<Link>> {
        let space = &self.space;
        let mut links = Vec::new();
        for message in header.of_kind(MSG_SYMBOL_TABLE) {
            let mut cursor = Cursor::new(message.body, message.offset);
            let btree = cursor.addr(space.offset_size)?;
            let heap_address = cursor.addr(space.offset_size)?;
            let heap = LocalHeap::parse(space, heap_address)?;
            for entry in btree1::group_entries(space, btree)? {
                let name = heap.string(space, entry.name_offset)?;
                walk.entry(name.len())?;
                links.push(Link {
                    name: String::from_utf8_lossy(name).into_owned(),
                    target: LinkTarget::Hard(entry.header),
                    creation_order: None,
                });
            }
        }
        let mut new_style = Vec::new();
        for message in header.of_kind(MSG_LINK) {
            let link = link::parse_link(message.body, message.offset, space.offset_size)?;
            walk.entry(link.name.len())?;
            new_style.push(link);
        }
        let mut ordered_by_index = false;
        if let Some(message) = header.first(MSG_LINK_INFO) {
            let info = link::parse_link_info(message.body, message.offset, space.offset_size)?;
            if info.heap != UNDEFINED_ADDR {
                walk.fractal_heap(space, info.heap)?;
                let (index, record_type) = match info.order_index {
                    Some(order) if order != UNDEFINED_ADDR => (order, 6u8),
                    _ => (info.name_index, 5u8),
                };
                ordered_by_index = record_type == 6;
                // Type 5: name hash (u32) + heap ID; type 6: creation order
                // (u64) + heap ID.
                let id_at = if record_type == 5 { 4 } else { 8 };
                let mut dense = Vec::new();
                btree2::for_each_record(space, index, &[record_type], id_at + 1, &mut |record| {
                    if dense.len() >= MAX_GROUP_ENTRIES {
                        return Err(limit(format!(
                            "HDF5 group has more than {MAX_GROUP_ENTRIES} entries (limit)"
                        )));
                    }
                    let id = record
                        .get(id_at..)
                        .ok_or_else(|| invalid(0, "link index record shorter than its heap ID"))?;
                    let Walk {
                        fractal, heap_work, ..
                    } = &mut *walk;
                    let heap = fractal
                        .get(&info.heap)
                        .ok_or_else(|| invalid(0, "fractal heap cache"))?;
                    let object = heap.object(space, id, heap_work)?;
                    let link = link::parse_link(&object, 0, space.offset_size)?;
                    walk.entry(link.name.len())?;
                    dense.push(link);
                    Ok(())
                })?;
                new_style.extend(dense);
            }
        }
        if !ordered_by_index {
            if new_style.iter().all(|link| link.creation_order.is_some()) {
                new_style.sort_by_key(|link| link.creation_order);
            } else {
                new_style.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
            }
        }
        links.extend(new_style);
        if links.len() > MAX_GROUP_ENTRIES {
            return Err(limit(format!(
                "HDF5 group has more than {MAX_GROUP_ENTRIES} entries (limit)"
            )));
        }
        Ok(links)
    }

    /// Attributes of one object: compact messages, then dense storage.
    fn read_attributes(
        &self,
        header: &ObjectHeader<'a>,
        walk: &mut Walk<'a>,
        cache: &mut CommittedTypes,
    ) -> Result<Vec<Attribute>> {
        let space = &self.space;
        let mut attributes = Vec::new();
        let mut compact_tracked = true;
        for message in header.of_kind(MSG_ATTRIBUTE) {
            if message.flags & FLAG_SHARED != 0 {
                return Err(unsupported("shared attribute messages (SOHM)"));
            }
            if attributes.len() >= MAX_ATTRIBUTES_PER_OBJECT {
                return Err(limit(format!(
                    "HDF5 object has more than {MAX_ATTRIBUTES_PER_OBJECT} attributes (limit)"
                )));
            }
            let mut attribute = self.parse_attribute(message.body, message.offset, walk, cache)?;
            attribute.creation_order = message.creation_order.map(u64::from);
            compact_tracked &= attribute.creation_order.is_some();
            attributes.push(attribute);
        }
        if compact_tracked && attributes.len() > 1 {
            attributes.sort_by_key(|attribute| attribute.creation_order);
        }
        if let Some(message) = header.first(MSG_ATTRIBUTE_INFO) {
            let info = link::parse_attribute_info(message.body, message.offset, space.offset_size)?;
            if info.heap != UNDEFINED_ADDR {
                let id_len = walk.fractal_heap(space, info.heap)?.id_len();
                let (index, record_type) = match info.order_index {
                    Some(order) if order != UNDEFINED_ADDR => (order, 9u8),
                    _ => (info.name_index, 8u8),
                };
                let mut dense: Vec<Attribute> = Vec::new();
                let min_record = id_len.saturating_add(5);
                btree2::for_each_record(space, index, &[record_type], min_record, &mut |record| {
                    if attributes.len() + dense.len() >= MAX_ATTRIBUTES_PER_OBJECT {
                        return Err(limit(format!(
                            "HDF5 object has more than {MAX_ATTRIBUTES_PER_OBJECT} attributes (limit)"
                        )));
                    }
                    // Types 8 and 9: heap ID, message flags (u8), creation
                    // order (u32)[, name hash (u32)].
                    let mut cursor = Cursor::new(record, 0);
                    let id = cursor.take(id_len)?;
                    let flags = cursor.u8()?;
                    let order = cursor.u32()?;
                    if flags & FLAG_SHARED != 0 {
                        return Err(unsupported("shared attribute messages (SOHM)"));
                    }
                    let Walk {
                        fractal, heap_work, ..
                    } = &mut *walk;
                    let heap = fractal
                        .get(&info.heap)
                        .ok_or_else(|| invalid(0, "fractal heap cache"))?;
                    let object = heap.object(space, id, heap_work)?;
                    let mut attribute = self.parse_attribute(&object, 0, walk, cache)?;
                    attribute.creation_order = Some(u64::from(order));
                    dense.push(attribute);
                    Ok(())
                })?;
                if record_type == 8 {
                    // The name index lists by name hash. netCDF-C lists
                    // attributes by creation index (H5Aiterate with
                    // H5_INDEX_CRT_ORDER), which HDF5 sorts out of the
                    // name index. A tracking object numbers them 0, 1, 2,
                    // ...; an untracking one stores 0xFFFF in every record,
                    // and the sort leaves them in name-index order.
                    dense.sort_by_key(|attribute| attribute.creation_order);
                    if !info.tracked {
                        for attribute in &mut dense {
                            attribute.creation_order = None;
                        }
                    }
                }
                attributes.extend(dense);
            }
        }
        if attributes.len() > MAX_ATTRIBUTES_PER_OBJECT {
            return Err(limit(format!(
                "HDF5 object has more than {MAX_ATTRIBUTES_PER_OBJECT} attributes (limit)"
            )));
        }
        Ok(attributes)
    }

    /// Attribute message: version (1-3), flags (v2+: bit 0 shared datatype,
    /// bit 1 shared dataspace), name size (u16), datatype size (u16),
    /// dataspace size (u16), [name character set (u8), v3], name, datatype,
    /// dataspace (each padded to 8 bytes in version 1), data.
    fn parse_attribute(
        &self,
        body: &[u8],
        offset: usize,
        walk: &mut Walk<'a>,
        cache: &mut CommittedTypes,
    ) -> Result<Attribute> {
        let mut cursor = Cursor::new(body, offset);
        let version = cursor.u8()?;
        if !(1..=3).contains(&version) {
            return Err(invalid(
                offset,
                format!("attribute version {version} unsupported"),
            ));
        }
        let flags = cursor.u8()?;
        let name_size = usize::from(cursor.u16()?);
        let datatype_size = usize::from(cursor.u16()?);
        let dataspace_size = usize::from(cursor.u16()?);
        if version == 3 {
            cursor.skip(1)?;
        }
        let padded = |len: usize| {
            if version == 1 {
                len.div_ceil(8) * 8
            } else {
                len
            }
        };
        let name_bytes = cursor.take(padded(name_size))?;
        let name = name_bytes[..name_size.min(name_bytes.len())]
            .split(|byte| *byte == 0)
            .next()
            .map(String::from_utf8_lossy)
            .unwrap_or_default()
            .into_owned();
        walk.entry(name.len())?;
        let datatype_at = offset + cursor.pos();
        let datatype_bytes = cursor.take(padded(datatype_size))?;
        let dataspace_at = offset + cursor.pos();
        let dataspace_bytes = cursor.take(padded(dataspace_size))?;
        let shared_type = version >= 2 && flags & 0x01 != 0;
        if version >= 2 && flags & 0x02 != 0 {
            return Err(unsupported("shared attribute dataspaces (SOHM)"));
        }
        let (datatype, _) = self.message_datatype(
            &datatype_bytes[..datatype_size.min(datatype_bytes.len())],
            datatype_at,
            if shared_type { FLAG_SHARED } else { 0 },
            cache,
        )?;
        let dataspace: Dataspace = dataspace::parse(
            &dataspace_bytes[..dataspace_size.min(dataspace_bytes.len())],
            dataspace_at,
            self.space.length_size,
        )?;
        let count = dataspace.element_count()?;
        let bytes = checked_bytes(count, datatype.size(), "HDF5 attribute")?;
        if bytes > MAX_ATTRIBUTE_BYTES {
            return Err(limit(format!(
                "HDF5 attribute requires {bytes} bytes (limit {MAX_ATTRIBUTE_BYTES})"
            )));
        }
        walk.attribute_bytes = walk.attribute_bytes.saturating_add(bytes);
        if walk.attribute_bytes > MAX_FILE_ATTRIBUTE_BYTES {
            return Err(limit(format!(
                "HDF5 attributes exceed {MAX_FILE_ATTRIBUTE_BYTES} decoded bytes per file (limit)"
            )));
        }
        let data = body.get(cursor.pos()..).unwrap_or_default();
        let mut decoder = Decoder::new(self.space, &mut walk.heaps);
        decoder.vlen_budget = MAX_ATTRIBUTE_BYTES;
        let values = decoder.decode(&datatype, data, count)?;
        walk.attribute_bytes = walk
            .attribute_bytes
            .saturating_add(MAX_ATTRIBUTE_BYTES - decoder.vlen_budget);
        Ok(Attribute {
            name,
            datatype,
            dims: dataspace.dims_usize(),
            null: dataspace.null,
            values,
            creation_order: None,
        })
    }
}

/// Address of the object a shared-message reference points at: version 1
/// (type, 6 reserved bytes, a length-size field, address), version 2 (type,
/// address) or version 3 (type 2 = committed object header address; type 1
/// = shared message heap, unsupported).
fn shared_address(body: &[u8], offset: usize, space: &Space<'_>) -> Result<u64> {
    let mut cursor = Cursor::new(body, offset);
    let version = cursor.u8()?;
    let kind = cursor.u8()?;
    match version {
        1 => {
            cursor.skip(6)?;
            cursor.skip(space.length_size)?;
            cursor.addr(space.offset_size)
        }
        2 => cursor.addr(space.offset_size),
        3 if kind == 2 => cursor.addr(space.offset_size),
        3 if kind == 1 => Err(unsupported("shared object header messages (SOHM)")),
        _ => Err(invalid(
            offset,
            format!("shared message version {version} type {kind}"),
        )),
    }
}

/// Fill value message versions 1-3.
fn parse_fill(body: &[u8], offset: usize) -> Result<Option<Vec<u8>>> {
    let mut cursor = Cursor::new(body, offset);
    let version = cursor.u8()?;
    let defined = match version {
        1 => {
            cursor.skip(2)?;
            cursor.u8()?;
            true
        }
        2 => {
            cursor.skip(2)?;
            cursor.u8()? != 0
        }
        3 => {
            let flags = cursor.u8()?;
            flags & 0x20 != 0
        }
        other => {
            return Err(invalid(
                offset,
                format!("fill value message version {other}"),
            ));
        }
    };
    if !defined || cursor.remaining() < 4 {
        return Ok(None);
    }
    let size = cursor.u32()? as usize;
    if size == 0 {
        return Ok(None);
    }
    Ok(Some(cursor.take(size)?.to_vec()))
}

/// Old fill value message: size (u32) + value.
fn parse_fill_old(body: &[u8], offset: usize) -> Result<Option<Vec<u8>>> {
    let mut cursor = Cursor::new(body, offset);
    let size = cursor.u32()? as usize;
    if size == 0 {
        return Ok(None);
    }
    Ok(Some(cursor.take(size)?.to_vec()))
}

fn checked_bytes(count: usize, element: usize, context: &str) -> Result<usize> {
    let bytes = count
        .checked_mul(element)
        .ok_or_else(|| invalid(0, format!("{context} byte-size overflow")))?;
    if bytes > MAX_DATASET_BYTES {
        return Err(limit(format!(
            "{context} requires {bytes} bytes (limit {MAX_DATASET_BYTES})"
        )));
    }
    Ok(bytes)
}

/// Bytes per element of the decoded representation (an upper bound used to
/// cap the converted size before reading).
fn decoded_width(datatype: &Datatype) -> usize {
    match datatype {
        Datatype::Integer { size, .. } | Datatype::Bitfield { size, .. } => {
            if matches!(size, 1 | 2 | 4 | 8) {
                *size
            } else {
                8
            }
        }
        Datatype::VarLenString { .. } | Datatype::VarLenSequence { .. } => 24,
        Datatype::Reference { .. } => 16,
        other => other.size().max(1),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::H5File;

    thread_local! {
        /// Committed datatype parses on this thread.
        pub(super) static COMMITTED_PARSES: Cell<usize> = const { Cell::new(0) };
    }

    /// Every plane's what gain/offset/nodata/undetect of the fixture (32
    /// attributes) use one committed datatype whose object header carries
    /// 30 attributes: it is parsed once per open, not once per attribute
    /// (tools/derive_hdf5_edge.py).
    #[test]
    fn committed_datatype_is_parsed_once_per_file() {
        let bytes = recast_radar_testdata::bytes("odim-dkrom-20260820-1130-pvol-h5edge-len4")
            .unwrap_or_else(|err| panic!("{err}"));
        COMMITTED_PARSES.with(|count| count.set(0));
        let file = H5File::open(&bytes).unwrap_or_else(|err| panic!("{err}"));
        let users = file
            .objects()
            .flat_map(|(_, object)| object.attributes())
            .filter(|attribute| attribute.name() == "gain")
            .count();
        assert_eq!(users, 16);
        assert_eq!(COMMITTED_PARSES.with(Cell::get), 1);
        for path in ["/dataset1/data1/data", "/dataset2/data8/data"] {
            file.dataset(path).unwrap_or_else(|err| panic!("{err}"));
        }
        assert_eq!(COMMITTED_PARSES.with(Cell::get), 1);
    }
}

#[cfg(test)]
mod limit_tests {
    use std::collections::HashMap;

    use super::{H5File, ObjectKind, OpenOptions, Walk};
    use crate::checksum::lookup3;
    use crate::heap::GlobalHeaps;
    use crate::limits::{MAX_FILE_ENTRIES, MAX_FILE_NAME_BYTES};
    use crate::{Error, Values};

    /// Superblock v3 behind a 512-byte user block, 8-byte offsets and
    /// lengths; dense links with name (type 5) and creation-order (type 6)
    /// indexes, dense attributes (8, 9), huge heap objects (1) and v2
    /// B-tree chunk indexes (10, 11) (`tools/derive_hdf5_latest.py`).
    const H5LATEST: &str = "odim-dkrom-20260820-1130-pvol-h5latest-trim";
    const BASE: usize = 512;
    /// v2 B-tree header bytes before the checksum with 8-byte offsets and
    /// lengths: signature, version, type, node size (u32), record size
    /// (u16), depth (u16), split and merge percents, root (O), root records
    /// (u16), total records (L).
    const BTHD_COVERED: usize = 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + 8 + 2 + 8;

    fn unverified() -> OpenOptions {
        OpenOptions::default().with_metadata_checksums(false)
    }

    /// The fixture's bytes and the offset and record type of each of its v2
    /// B-tree headers.
    fn h5latest() -> (Vec<u8>, Vec<(usize, u8)>) {
        let bytes = recast_radar_testdata::bytes(H5LATEST).unwrap_or_else(|err| panic!("{err}"));
        let headers = bytes
            .windows(4)
            .enumerate()
            .filter(|(_, window)| *window == b"BTHD")
            .map(|(at, _)| (at, bytes[at + 5]))
            .collect();
        (bytes, headers)
    }

    /// The lookup3 checksum of `covered` bytes at `start`, little-endian,
    /// for storing after them.
    fn seal(bytes: &[u8], start: usize, covered: usize) -> [u8; 4] {
        lookup3(&bytes[start..start + covered]).to_le_bytes()
    }

    /// Open, then read every dataset; `Ok` means the whole file read.
    fn read_all(bytes: &[u8], options: OpenOptions) -> Result<(), Error> {
        let file = H5File::open_with(bytes, options)?;
        let paths: Vec<String> = file.objects().map(|(path, _)| path.to_owned()).collect();
        for path in paths {
            if file.object(&path).map(|object| object.kind()) == Some(ObjectKind::Dataset) {
                file.dataset(&path)?;
            }
        }
        Ok(())
    }

    /// The creation-order link index (record type 6: creation order (u64)
    /// and a 7-byte heap ID) rewritten with 4-byte records, its header and
    /// leaf checksums recomputed so a checksum-verifying reader accepts
    /// them. The walk sliced each record at byte 8 and panicked; the record
    /// size is now checked against the record type before any record is
    /// read.
    #[test]
    fn short_link_index_records_are_an_error_not_a_panic() {
        let (original, headers) = h5latest();
        let (at, _) = *headers
            .iter()
            .find(|(_, kind)| *kind == 6)
            .expect("h5latest has a creation-order link index");
        assert_eq!(
            u16::from_le_bytes([original[at + 10], original[at + 11]]),
            15
        );
        assert_eq!(
            u16::from_le_bytes([original[at + 12], original[at + 13]]),
            0
        );
        let root = u64::from_le_bytes(original[at + 16..at + 24].try_into().unwrap()) as usize;
        let records = usize::from(u16::from_le_bytes([original[at + 24], original[at + 25]]));
        assert!(records > 0);
        let mut bytes = original.clone();
        bytes[at + 10..at + 12].copy_from_slice(&4u16.to_le_bytes());
        let sum = seal(&bytes, at, BTHD_COVERED);
        bytes[at + BTHD_COVERED..at + BTHD_COVERED + 4].copy_from_slice(&sum);
        let leaf = BASE + root;
        assert_eq!(&bytes[leaf..leaf + 4], b"BTLF");
        let content = 6 + records * 4;
        let sum = seal(&bytes, leaf, content);
        bytes[leaf + content..leaf + content + 4].copy_from_slice(&sum);
        let err = match H5File::open(&bytes) {
            Ok(_) => panic!("a link index of 4-byte records opened"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("need at least 9"), "{err}");
        // The untouched file opens.
        H5File::open(&original).unwrap();
    }

    /// Every v2 B-tree header of the fixture declaring each shorter record
    /// size and each other record type (header checksum recomputed), read
    /// with and without checksum verification: errors, never a panic.
    #[test]
    fn any_record_size_or_type_in_a_btree_header_is_safe() {
        let (original, headers) = h5latest();
        assert_eq!(headers.len(), 9);
        let mut tried = 0;
        for (at, kind) in headers {
            let size = u16::from_le_bytes([original[at + 10], original[at + 11]]);
            let variants = (0..size)
                .map(|size| (kind, size))
                .chain((0..=12u8).filter(|t| *t != kind).map(|t| (t, size)));
            for (record_type, record_size) in variants {
                let mut bytes = original.clone();
                bytes[at + 5] = record_type;
                bytes[at + 10..at + 12].copy_from_slice(&record_size.to_le_bytes());
                let sum = seal(&bytes, at, BTHD_COVERED);
                bytes[at + BTHD_COVERED..at + BTHD_COVERED + 4].copy_from_slice(&sum);
                for options in [OpenOptions::default(), unverified()] {
                    let _ = read_all(&bytes, options);
                }
                tried += 1;
            }
        }
        assert!(tried > 150, "{tried}");
    }

    /// With checksum verification off, a file whose stored checksums are
    /// wrong reads exactly like the original; with it on (the default), the
    /// same bytes are a checksum error.
    #[test]
    fn metadata_checksums_can_be_skipped() {
        let (original, headers) = h5latest();
        let reference = H5File::open(&original).unwrap();
        let mut bytes = original.clone();
        // Every B-tree header's stored checksum, and the superblock's
        // (signature, version, sizes, flags, four addresses).
        for (at, _) in headers {
            bytes[at + BTHD_COVERED] ^= 0xFF;
        }
        let superblock_covered = 8 + 1 + 1 + 1 + 1 + 4 * 8;
        bytes[BASE + superblock_covered] ^= 0xFF;
        assert!(matches!(H5File::open(&bytes), Err(Error::Checksum { .. })));
        let read = H5File::open_with(&bytes, unverified()).unwrap();
        assert_eq!(reference.objects().count(), read.objects().count());
        let mut datasets = 0;
        for ((path, object), (read_path, read_object)) in reference.objects().zip(read.objects()) {
            assert_eq!(path, read_path);
            assert_eq!(object.attributes(), read_object.attributes(), "{path}");
            if object.kind() == ObjectKind::Dataset {
                let a: Values = reference.dataset(path).unwrap().values;
                let b: Values = read.dataset(path).unwrap().values;
                assert_eq!(a, b, "{path}");
                datasets += 1;
            }
        }
        assert!(datasets > 0);
    }

    fn walk() -> Walk<'static> {
        Walk {
            heaps: GlobalHeaps::default(),
            attribute_bytes: 0,
            path_bytes: 0,
            entries: 0,
            name_bytes: 0,
            fractal: HashMap::new(),
            heap_work: 0,
        }
    }

    /// Links and attributes count against one per-file total, names against
    /// one per-file byte total, whatever their storage (an attribute with a
    /// null dataspace has no value bytes but still counts).
    #[test]
    fn links_attributes_and_names_have_per_file_limits() {
        let mut entries = walk();
        for _ in 0..MAX_FILE_ENTRIES {
            entries.entry(0).unwrap();
        }
        assert!(matches!(entries.entry(0), Err(Error::LimitExceeded(_))));
        let mut names = walk();
        names.entry(MAX_FILE_NAME_BYTES).unwrap();
        assert!(matches!(names.entry(1), Err(Error::LimitExceeded(_))));
    }
}
