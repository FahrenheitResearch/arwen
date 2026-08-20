//! Reader for the RWGOLD01 mesh-golden container, plus the canonical
//! edge/vertex enumeration the goldens are expressed in.
//!
//! Test-only. No netCDF, no network, no dependency outside std: a golden has to
//! be loadable in an offline `cargo test --locked --offline`, which is how this
//! crate is built.
//!
//! The container is described in tests/goldens/make_mesh_goldens.py. Layout:
//!   magic "RWGOLD01" (8) | u32 version | u32 n_entries
//!   n_entries x 104-byte directory records, then 8-byte-aligned payload.
//!   record: name[64] NUL-padded | u8 dtype (1=i32, 2=f64) | u8 ndim | u16 pad
//!           | u32 shape[4] | u64 offset | u64 nbytes | 4 pad
//! Everything is little-endian.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"RWGOLD01";
const VERSION: u32 = 1;
const ENTRY_BYTES: usize = 104;
const NAME_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    I32,
    F64,
}

pub struct Entry {
    pub dtype: DType,
    pub shape: Vec<usize>,
    off: usize,
    nbytes: usize,
}

pub struct Rwg {
    blob: Vec<u8>,
    entries: HashMap<String, Entry>,
    order: Vec<String>,
}

fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b.try_into().unwrap())
}
fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes(b.try_into().unwrap())
}

impl Rwg {
    pub fn load(path: impl AsRef<Path>) -> Rwg {
        let path = path.as_ref();
        let blob = std::fs::read(path)
            .unwrap_or_else(|e| panic!("mesh golden {} could not be read: {e}", path.display()));
        assert!(blob.len() >= 16, "{}: shorter than a header", path.display());
        assert_eq!(&blob[..8], MAGIC, "{}: not an RWGOLD01 container", path.display());
        let version = u32le(&blob[8..12]);
        assert_eq!(version, VERSION, "{}: container version {version}", path.display());
        let n = u32le(&blob[12..16]) as usize;

        let mut entries = HashMap::new();
        let mut order = Vec::with_capacity(n);
        for i in 0..n {
            let base = 16 + i * ENTRY_BYTES;
            let rec = &blob[base..base + ENTRY_BYTES];
            let name_end = rec[..NAME_BYTES].iter().position(|&c| c == 0).unwrap_or(NAME_BYTES);
            let name = String::from_utf8(rec[..name_end].to_vec()).expect("entry name is not utf-8");
            let dtype = match rec[64] {
                1 => DType::I32,
                2 => DType::F64,
                other => panic!("{name}: unknown dtype code {other}"),
            };
            let ndim = rec[65] as usize;
            assert!((1..=4).contains(&ndim), "{name}: ndim {ndim}");
            let shape: Vec<usize> =
                (0..ndim).map(|d| u32le(&rec[68 + 4 * d..72 + 4 * d]) as usize).collect();
            let off = u64le(&rec[84..92]) as usize;
            let nbytes = u64le(&rec[92..100]) as usize;

            let elem = if dtype == DType::I32 { 4 } else { 8 };
            let want: usize = shape.iter().product::<usize>() * elem;
            assert_eq!(nbytes, want, "{name}: declared {nbytes} B, shape implies {want} B");
            assert!(off + nbytes <= blob.len(), "{name}: slab runs past end of file");
            assert_eq!(off % 8, 0, "{name}: slab is not 8-byte aligned");

            order.push(name.clone());
            assert!(
                entries.insert(name.clone(), Entry { dtype, shape, off, nbytes }).is_none(),
                "{name}: duplicate entry"
            );
        }
        Rwg { blob, entries, order }
    }

    pub fn names(&self) -> &[String] {
        &self.order
    }

    pub fn entry(&self, name: &str) -> &Entry {
        self.entries
            .get(name)
            .unwrap_or_else(|| panic!("mesh golden has no entry {name:?}"))
    }

    pub fn shape(&self, name: &str) -> &[usize] {
        &self.entry(name).shape
    }

    pub fn i32(&self, name: &str) -> Vec<i32> {
        let e = self.entry(name);
        assert_eq!(e.dtype, DType::I32, "{name}: expected i32");
        self.blob[e.off..e.off + e.nbytes]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    pub fn f64(&self, name: &str) -> Vec<f64> {
        let e = self.entry(name);
        assert_eq!(e.dtype, DType::F64, "{name}: expected f64");
        self.blob[e.off..e.off + e.nbytes]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Raw slab bytes, for byte-exact round-trip checks.
    pub fn raw(&self, name: &str) -> &[u8] {
        let e = self.entry(name);
        &self.blob[e.off..e.off + e.nbytes]
    }
}

pub fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("goldens")
}

/// The two published meshes, by the name the golden files carry.
pub const MESHES: [&str; 2] = ["x1.40962", "x4.163842"];

pub fn load_mesh(tag: &str) -> Rwg {
    Rwg::load(golden_dir().join(format!("{tag}.mesh.rwg")))
}

// ------------------------------------------------------------ canonical numbering

/// Canonical edge list: every unordered cell pair that shares an edge, each as
/// `[a, b]` with `a < b`, sorted lexicographically.
///
/// This is a pure function of the cell centres (via the cellsOnCell ring), which
/// is exactly why the goldens use it: a generator invents its own edge numbering,
/// so the file's numbering is not something a correct generator has to match.
pub fn canonical_edges(offsets: &[i32], values: &[i32]) -> Vec<[i32; 2]> {
    let n_cells = offsets.len() - 1;
    let mut pairs: Vec<[i32; 2]> = Vec::with_capacity(values.len());
    for i in 0..n_cells {
        for k in offsets[i] as usize..offsets[i + 1] as usize {
            let (a, b) = (i as i32, values[k]);
            pairs.push([a.min(b), a.max(b)]);
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Canonical vertex list: every sorted cell triple that meets at a vertex,
/// sorted lexicographically. Also a pure function of the cell centres.
///
/// The triple for ring slot `j` of cell `i` is `{i, ring[j-1], ring[j]}` --
/// measured on every cell of both published meshes with zero violations, while
/// the `{i, ring[j], ring[j+1]}` alternative is violated on every slot.
pub fn canonical_vertices(offsets: &[i32], values: &[i32]) -> Vec<[i32; 3]> {
    let n_cells = offsets.len() - 1;
    let mut tris: Vec<[i32; 3]> = Vec::with_capacity(values.len());
    for i in 0..n_cells {
        let lo = offsets[i] as usize;
        let hi = offsets[i + 1] as usize;
        let deg = hi - lo;
        for j in 0..deg {
            let prev = values[lo + (j + deg - 1) % deg];
            let cur = values[lo + j];
            let mut t = [i as i32, prev, cur];
            t.sort_unstable();
            tris.push(t);
        }
    }
    tris.sort_unstable();
    tris.dedup();
    tris
}

/// Index of `key` in a sorted canonical table, or None.
pub fn lookup<T: Ord>(table: &[T], key: &T) -> Option<usize> {
    table.binary_search(key).ok()
}
