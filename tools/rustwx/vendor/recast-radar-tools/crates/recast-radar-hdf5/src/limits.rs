//! Resource limits (documented in the crate-level `# Limits` table).

/// Deepest group nesting walked from the root group.
pub(crate) const MAX_GROUP_DEPTH: usize = 16;
/// Most named objects (paths) indexed per file.
pub(crate) const MAX_OBJECTS: usize = 1 << 14;
/// Total bytes of all indexed path names.
pub(crate) const MAX_PATH_BYTES: usize = 64 << 20;
/// Longest link (object) name.
pub(crate) const MAX_LINK_NAME_BYTES: usize = 1 << 16;
/// Most nodes visited in one B-tree walk (v1 or v2).
pub(crate) const MAX_BTREE_NODES: usize = 1 << 16;
/// Deepest v2 B-tree.
pub(crate) const MAX_BTREE2_DEPTH: usize = 32;
/// Largest v2 B-tree node (the HDF5 library writes 512 and 2048 bytes).
pub(crate) const MAX_BTREE2_NODE_BYTES: usize = 1 << 20;
/// Most header blocks (chunk 0 plus continuations) per object header.
pub(crate) const MAX_HEADER_BLOCKS: usize = 1 << 10;
/// Most messages per object header.
pub(crate) const MAX_OBJECT_MESSAGES: usize = 4096;
/// Most message bytes per object header, and per header block.
pub(crate) const MAX_OBJECT_MESSAGE_BYTES: usize = 64 << 20;
/// Most links in one group, and records in one v2 B-tree.
pub(crate) const MAX_GROUP_ENTRIES: usize = 1 << 20;
/// Most attributes on one object.
pub(crate) const MAX_ATTRIBUTES_PER_OBJECT: usize = 1 << 16;
/// Most chunks per dataset.
pub(crate) const MAX_DATA_CHUNKS: usize = 1 << 18;
/// Highest dataspace rank.
pub(crate) const MAX_DATASPACE_RANK: usize = 32;
/// Largest current dataspace dimension.
pub(crate) const MAX_DATASPACE_DIM: u64 = 100 * 1024 * 1024;
/// Largest dataset, stored and after type conversion (each).
pub(crate) const MAX_DATASET_BYTES: usize = 256 << 20;
/// Largest single attribute value.
pub(crate) const MAX_ATTRIBUTE_BYTES: usize = 16 << 20;
/// Most decoded attribute bytes per file (all objects together).
pub(crate) const MAX_FILE_ATTRIBUTE_BYTES: usize = 256 << 20;
/// Most links and attributes decoded per file (all objects together; an
/// index that several objects share counts once per object).
pub(crate) const MAX_FILE_ENTRIES: usize = 1 << 20;
/// Most bytes of link and attribute names decoded per file.
pub(crate) const MAX_FILE_NAME_BYTES: usize = 64 << 20;
/// Most fractal-heap block bytes checksummed or inflated per file.
pub(crate) const MAX_FILE_HEAP_BYTES: usize = 1 << 30;
/// Most filters per pipeline.
pub(crate) const MAX_FILTERS: usize = 32;
/// Most client values per filter.
pub(crate) const MAX_FILTER_VALUES: usize = 1024;
/// Deepest datatype nesting (compound, array, enum, variable-length).
pub(crate) const MAX_DATATYPE_DEPTH: usize = 16;
/// Largest fractal-heap direct block or managed object.
pub(crate) const MAX_HEAP_BLOCK_BYTES: usize = 64 << 20;
/// Most fractal-heap indirect-block levels followed for one object.
pub(crate) const MAX_HEAP_DEPTH: usize = 64;
/// Most bytes read from global heap collections for one dataset or
/// attribute (variable-length data).
pub(crate) const MAX_VLEN_BYTES: usize = 256 << 20;
