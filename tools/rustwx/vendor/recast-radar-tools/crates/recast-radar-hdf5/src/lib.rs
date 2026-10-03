//! Pure-Rust HDF5 reader and writer for weather radar files.
//!
//! Written from The HDF Group's "HDF5 File Format Specification Version
//! 3.0" (<https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html>)
//! and checked against h5py on real ODIM_H5, netCDF-4 CfRadial 1 and
//! CfRadial 2 files. No C library, no `unsafe`.
//!
//! # Coverage
//!
//! - Superblocks versions 0-3, including a user block before the signature.
//! - Object headers version 1 and 2 ("OHDR"/"OCHK" with lookup3 checksums).
//! - Groups: old-style symbol tables (v1 B-tree + local heap); new-style
//!   compact link messages; dense links in a fractal heap indexed by a v2
//!   B-tree (name index, or creation-order index when present); hard, soft
//!   and external links.
//! - Attributes: compact attribute messages versions 1-3, and dense
//!   attribute storage (fractal heap + v2 B-tree name or creation-order
//!   index).
//! - Fractal heaps: managed objects in direct and indirect blocks (checksums
//!   verified, I/O filters applied), huge objects (directly addressed or
//!   through the huge-object v2 B-tree), tiny objects.
//! - Data layouts: compact, contiguous, chunked (layout messages 1-5;
//!   version 5 is HDF5 2.0's).
//! - Chunk indexes: v1 B-tree, single chunk, implicit, fixed array (paged
//!   and unpaged), extensible array (index, secondary and data blocks, paged
//!   data blocks), v2 B-tree (record types 10 and 11).
//! - Filters: deflate, shuffle, Fletcher-32. Others (szip, LZF, n-bit,
//!   scale-offset, third-party) are [`Error::UnsupportedFilter`] errors.
//! - Datatypes: integers of 1-8 bytes (bit offset and precision honoured),
//!   IEEE binary32/64, fixed and variable-length strings, bit fields,
//!   opaque, compound, object references, enumerations, variable-length
//!   sequences, arrays, and committed (named) datatypes. Values keep their
//!   stored type ([`Values`]); anything else comes back as raw bytes.
//! - Fill values (fill value messages old and new) for unwritten storage.
//!
//! Not supported (typed [`Error::Unsupported`] errors): shared object header
//! messages (SOHM), virtual datasets, external raw data files.
//!
//! Checked against h5py on real radar data (`tests/h5py_goldens.rs`): the
//! corpus files as their producers wrote them, and HDF5-library containers
//! derived from real ODIM files (`tools/derive_hdf5_latest.py`,
//! `tools/derive_hdf5_edge.py`) for what no producer writes: superblock v3
//! behind a user block, every version-4 chunk index, paged fixed and
//! extensible arrays (sparsely written too), dense links and attributes with
//! creation-order indexes, huge heap objects through the huge-object B-tree,
//! a deflated fractal heap, committed datatypes, 4-byte addresses and 4-byte
//! lengths. The typed refusal of other filters is tested on the X-SAPR
//! netCDF-4 file with its reflectivity also stored szip-compressed (by
//! netCDF-C with libaec) and LZF-compressed (by h5py)
//! (`tools/derive_hdf5_filters.py`, `tests/unsupported_filters.rs`). No file
//! the HDF5 library writes has tiny fractal heap objects or directly
//! addressed huge objects (its heap IDs for links and attributes, 7 and 8
//! bytes, are too short for either): those two paths are written from the
//! specification and untested.
//!
//! # Writing
//!
//! [`write::Writer`] builds HDF5 1.8-format files (version 2 superblock and
//! object headers, compact new-style groups and attributes, contiguous,
//! compact and chunked datasets with shuffle and deflate, variable-length
//! strings, object references and the dimension scale attributes netCDF-4
//! uses). Output is deterministic and reads back with h5py, netCDF-C and
//! this crate's reader.
//!
//! # netCDF-4
//!
//! [`netcdf4::NcFile`] rebuilds the netCDF-4 data model of an HDF5 file the
//! way netCDF-C reads it back: groups, dimensions from dimension scales
//! (unlimited ones included, phony dimensions for datasets without scales),
//! variables with their dimensions and netCDF types, attributes without the
//! ones netCDF-C hides, and the root `_NCProperties`.
//!
//! # Usage
//!
//! ```no_run
//! # fn main() -> Result<(), recast_radar_hdf5::Error> {
//! let bytes = std::fs::read("volume.h5").map_err(|e| recast_radar_hdf5::Error::Unsupported(e.to_string()))?;
//! let file = recast_radar_hdf5::H5File::open(&bytes)?;
//! for name in file.child_names("/") {
//!     println!("{name}");
//! }
//! if let Some(object) = file.attr("/what", "object").and_then(|a| a.as_str()) {
//!     println!("ODIM object {object}");
//! }
//! let data = file.dataset("/dataset1/data1/data")?;
//! println!("{:?} {} elements", data.dims, data.values.len());
//! # Ok(())
//! # }
//! ```
//!
//! # Caching
//!
//! [`H5File::open`] walks the group tree once, parses every object header
//! once and decodes every attribute once. Attribute and link lookups are map
//! lookups afterwards; dataset reads parse only the dataset's own messages
//! (already in memory) and its chunk index.
//!
//! # Limits
//!
//! Every size a file declares is checked before it is used:
//!
//! | Structure | Limit |
//! |---|---|
//! | Group nesting depth | 16 |
//! | Objects (paths) indexed per file | 16,384 |
//! | Bytes of all indexed paths | 64 MiB |
//! | Link name | 65,536 bytes |
//! | B-tree nodes per walk (v1 and v2) | 65,536 |
//! | v2 B-tree depth / node size | 32 / 1 MiB |
//! | v2 B-tree node records | the node's capacity (`H5B2__hdr_init`) |
//! | Records per v2 B-tree, links per group | 1,048,576 |
//! | Attributes per object | 65,536 |
//! | Links and attributes per file (an index several objects share counts once per object) | 1,048,576 |
//! | Bytes of link and attribute names per file | 64 MiB |
//! | Messages per object header | 4,096 |
//! | Message bytes per object header, and per header block | 64 MiB |
//! | Header continuation blocks per object | 1,024 |
//! | Dataspace rank / current dimension size | 32 / 104,857,600 |
//! | Dataset bytes, stored and after type conversion | 256 MiB each |
//! | Chunks per dataset / stored chunk size | 262,144 / 256 MiB |
//! | Inflated chunk | its declared chunk size (+64 bytes of checksums) |
//! | Attribute value | 16 MiB |
//! | Decoded attribute bytes per file | 256 MiB |
//! | Filters per pipeline / client values per filter | 32 / 1,024 |
//! | Datatype nesting depth | 16 |
//! | Fractal heap block or object | 64 MiB |
//! | Fractal heap indirect-block depth | 64 |
//! | Fractal heap bytes checksummed, inflated or indexed per file | 1 GiB |
//! | Variable-length bytes per dataset | 256 MiB |
//!
//! Every limit violation is an [`Error::LimitExceeded`] error. A v2
//! B-tree's record type and record size are checked against what its user
//! needs before any record is read.
//!
//! # Checksums
//!
//! [`H5File::open`] verifies every lookup3 metadata checksum, as the HDF5
//! library does. [`H5File::open_with`] and
//! [`OpenOptions::with_metadata_checksums`] can turn that off: the file
//! reads as far as its structures parse, with every limit above still
//! applied. The `hdf5`, `odim` and `cfradial` fuzz harnesses read each input
//! both ways, so mutated inputs reach the parsers behind the checksums.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
#![deny(missing_docs)]

mod btree1;
mod btree2;
mod bytes;
mod checksum;
mod chunks;
mod dataspace;
mod datatype;
mod error;
mod file;
mod filters;
mod fractal;
mod header;
mod heap;
mod layout;
mod limits;
mod link;
pub mod netcdf4;
mod space;
mod superblock;
mod values;
pub mod write;

pub use dataspace::UNLIMITED;
pub use datatype::{
    ByteOrder, CharSet, CompoundMember, Datatype, EnumMember, ReferenceKind, StringPadding,
};
pub use error::{Error, Result};
pub use file::{
    Attribute, ChunkLocation, Dataset, DatasetInfo, H5File, Object, ObjectKind, OpenOptions,
};
pub use filters::Filter;
pub use layout::{ChunkIndexKind, StorageLayout};
pub use link::{Link, LinkTarget};
pub use values::Values;

/// `true` when the buffer holds the HDF5 format signature where a
/// superblock may start: offset 0, or after a user block at 512, 1024,
/// 2048, ... bytes (HDF5 File Format Specification, section II), or after a
/// header of up to 64 KiB that a distributor has put in front of the file
/// (ECCC volume scans start with a text heading).
pub fn looks_like_hdf5_bytes(bytes: &[u8]) -> bool {
    superblock::signature_offset(bytes).is_some()
}

#[cfg(test)]
mod tests {
    /// Opened files can be shared across threads (dataset reads take
    /// `&self`).
    #[test]
    fn file_view_is_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<crate::H5File<'static>>();
    }
}
