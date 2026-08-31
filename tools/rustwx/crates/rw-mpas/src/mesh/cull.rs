//! Region culler: cut a limited-area mesh out of a global MPAS parent file.
//!
//! This is the production Rust arm of the regional mesh path. Its match target
//! is MPAS-Limited-Area v2.2 (commit edc556e17, 2025-04-01), the tool that
//! minted the pinned native reference culls, and the goal is BYTE IDENTITY
//! with that tool's output for the same region on the same parent: header,
//! variable order, attributes, masks, connectivity, coordinate subsets, and
//! the METIS graph file.
//!
//! The region request is DATA: the crate's existing [`Shape`] rows
//! (`cap` / `lat_lon_box` / `polygon`), the same rows `--spec` refinement
//! regions use. Adding a region is a JSON row, never a code path.
//!
//! ## The measured conventions this module reproduces
//!
//! Measured on the native culls of x1.40962 and x4.163842 (records of
//! 2026-08-25, five cull-measurement JSONs) and confirmed against the tool's
//! own source:
//!
//! * Exactly three variables are added: `bdyMaskCell`, `bdyMaskEdge`,
//!   `bdyMaskVertex` (int32, no attributes), defined FIRST, before the parent
//!   variables. On a parent that already carries them (a static file), the
//!   parent's copies are dropped and these take their place.
//! * Cell masks: 0 interior, rings 1..7 outward. Internally the tool marks
//!   interior 1 and rings 2..8, then subtracts 1 on write.
//! * Edge/vertex masks: MIN of the adjacent cell masks when every adjacent
//!   cell is marked, else MAX (so a rim element takes its one marked cell's
//!   value). Derived on the GLOBAL mesh before subsetting.
//! * Element order is the parent subset order, overall and within every mask
//!   class. Nothing is block-sorted; `indexTo*ID` is rewritten to a contiguous
//!   `1..N` and no parent-index map is stored.
//! * Connectivity entries pointing at culled elements become 0. Valid-slot
//!   zeros therefore appear ONLY on mask-7 elements, and only in
//!   `cellsOnCell`, `cellsOnEdge`, `edgesOnEdge` (inside the declared
//!   `nEdgesOnEdge` row length, which is NOT shrunk), `cellsOnVertex` and
//!   `edgesOnVertex`. `edgesOnCell`, `verticesOnCell` and `verticesOnEdge`
//!   acquire none, because every edge and vertex of a marked cell is marked.
//! * Global attributes: only `on_a_sphere` and `sphere_radius` survive, in
//!   that order, with the parent's own attribute type and bytes.
//! * Per-variable attributes: `units` then `long_name`, copied from the
//!   parent with the tool's exact partial-failure rule -- a variable with no
//!   `units` gets NEITHER attribute, even when `long_name` exists.
//! * Output format matches the parent's classic format (CDF-1/2/5).
//! * Euler characteristic `V - E + F = 1`: the region is a disk.
//!
//! ## One deliberate divergence from the native tool
//!
//! The tool reindexes with `map[field - 1]` in numpy, so a stored 0 reads
//! `map[N-1]` -- the map entry of the LAST global element. When the last
//! global element is outside the region that entry is 0 and the wrap is
//! invisible; when it is inside, the tool writes that element's regional index
//! into slots that mean "no neighbour". This module maps stored 0 to 0
//! always, and the receipt records whether the parent would have tripped the
//! wrap ([`CullReceipt::native_wrap_divergence`]). Both pinned parents have
//! their last cell, edge and vertex outside the region, so the outputs are
//! byte-identical there.

use std::io::Write as _;
use std::path::Path;

use serde::Serialize;

use crate::error::{MpasError, MpasResult};
use crate::mesh::density::Shape;

use rw_store::netcdf_classic::{
    NcAttr, NcAttrValue, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};

/// Boundary layers the native tool grows outward, spelled from its source
/// (`LimitedArea.num_boundary_layers = 8`): interior 1 plus rings 2..8,
/// written out as 0..7.
pub const NUM_BOUNDARY_LAYERS: i32 = 8;

const INSIDE: i32 = 1;
const UNMARKED: i32 = 0;

/// The eleven fields whose VALUES are element indices and are rewritten
/// through the renumber maps, exactly the native tool's `indexingFields`
/// table. The second column names which map rewrites the values.
const INDEXING_FIELDS: [(&str, ElemClass); 11] = [
    ("indexToCellID", ElemClass::Cell),
    ("indexToEdgeID", ElemClass::Edge),
    ("indexToVertexID", ElemClass::Vertex),
    ("cellsOnEdge", ElemClass::Cell),
    ("edgesOnCell", ElemClass::Edge),
    ("edgesOnEdge", ElemClass::Edge),
    ("cellsOnCell", ElemClass::Cell),
    ("verticesOnCell", ElemClass::Vertex),
    ("verticesOnEdge", ElemClass::Vertex),
    ("edgesOnVertex", ElemClass::Edge),
    ("cellsOnVertex", ElemClass::Cell),
];

/// Which element family a dimension or a map belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemClass {
    Cell,
    Edge,
    Vertex,
}

// ===========================================================================
// Raw classic-format reader
// ===========================================================================
//
// The cull is a byte-level subset: most variables are copied row-for-row with
// no numeric decode at all, which is what makes "coordinate fields match
// exactly" a construction property instead of a hope. `netcrust` reads
// everything through f64 promotion, so this private reader parses the classic
// header itself -- the same grammar `rw_store::netcdf_classic` writes, proven
// byte-for-byte against netCDF-C goldens -- and hands out raw slabs at the
// parent's own stored offsets.

mod raw {
    use crate::error::{MpasError, MpasResult};

    pub const NC_BYTE: u32 = 1;
    pub const NC_CHAR: u32 = 2;
    pub const NC_SHORT: u32 = 3;
    pub const NC_INT: u32 = 4;
    pub const NC_FLOAT: u32 = 5;
    pub const NC_DOUBLE: u32 = 6;

    pub fn type_size(t: u32) -> MpasResult<usize> {
        Ok(match t {
            NC_BYTE | NC_CHAR => 1,
            NC_SHORT => 2,
            NC_INT | NC_FLOAT => 4,
            NC_DOUBLE => 8,
            other => {
                return Err(MpasError::Refusal(format!(
                    "parent carries nc_type {other}, outside the classic type set; \
                     a slab of unknown width cannot be subset without corrupting \
                     every later variable"
                )))
            }
        })
    }

    #[derive(Debug, Clone)]
    pub struct RawDim {
        pub name: String,
        pub len: usize,
        pub unlimited: bool,
    }

    #[derive(Debug, Clone)]
    pub struct RawAttr {
        pub name: String,
        pub nc_type: u32,
        /// Value bytes, unpadded (`nelems * type_size`), exactly as stored.
        pub value: Vec<u8>,
    }

    #[derive(Debug, Clone)]
    pub struct RawVar {
        pub name: String,
        pub dimids: Vec<usize>,
        pub attrs: Vec<RawAttr>,
        pub nc_type: u32,
        pub begin: u64,
        pub is_record: bool,
        /// Unpadded bytes of one slab (whole array for fixed variables, one
        /// record's worth for record variables).
        pub slab_bytes: usize,
        /// Bytes one slab occupies on disk, padding included.
        pub slab_stride: u64,
    }

    pub struct RawNc {
        pub bytes: Vec<u8>,
        pub version: u8,
        pub numrecs: u64,
        pub dims: Vec<RawDim>,
        pub gattrs: Vec<RawAttr>,
        pub vars: Vec<RawVar>,
        pub record_stride: u64,
    }

    struct Cursor<'a> {
        b: &'a [u8],
        pos: usize,
        nonneg8: bool,
        offset8: bool,
    }

    impl<'a> Cursor<'a> {
        fn take(&mut self, n: usize) -> MpasResult<&'a [u8]> {
            if self.pos + n > self.b.len() {
                return Err(MpasError::Refusal(format!(
                    "classic header truncated at byte {}: wanted {n} more",
                    self.pos
                )));
            }
            let s = &self.b[self.pos..self.pos + n];
            self.pos += n;
            Ok(s)
        }
        fn u32(&mut self) -> MpasResult<u32> {
            Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
        }
        fn nonneg(&mut self) -> MpasResult<u64> {
            if self.nonneg8 {
                Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
            } else {
                Ok(self.u32()? as u64)
            }
        }
        fn offset(&mut self) -> MpasResult<u64> {
            if self.offset8 {
                Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
            } else {
                Ok(self.u32()? as u64)
            }
        }
        fn name(&mut self) -> MpasResult<String> {
            let len = self.nonneg()? as usize;
            let bytes = self.take(len)?.to_vec();
            let pad = (4 - (len % 4)) % 4;
            self.take(pad)?;
            String::from_utf8(bytes).map_err(|_| {
                MpasError::Refusal("classic header carries a non-UTF-8 name".to_string())
            })
        }
        fn attr_list(&mut self) -> MpasResult<Vec<RawAttr>> {
            let tag = self.u32()?;
            let n = self.nonneg()? as usize;
            if tag == 0 {
                if n != 0 {
                    return Err(MpasError::Refusal(
                        "classic header ABSENT attribute list with nonzero count".to_string(),
                    ));
                }
                return Ok(Vec::new());
            }
            if tag != 0x0C {
                return Err(MpasError::Refusal(format!(
                    "classic header expected NC_ATTRIBUTE, found tag 0x{tag:X}"
                )));
            }
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                let name = self.name()?;
                let nc_type = self.u32()?;
                let nelems = self.nonneg()? as usize;
                let width = type_size(nc_type)?;
                let raw_len = nelems * width;
                let value = self.take(raw_len)?.to_vec();
                let pad = (4 - (raw_len % 4)) % 4;
                self.take(pad)?;
                out.push(RawAttr {
                    name,
                    nc_type,
                    value,
                });
            }
            Ok(out)
        }
    }

    impl RawNc {
        pub fn open(path: &std::path::Path) -> MpasResult<Self> {
            let bytes = std::fs::read(path)?;
            if bytes.len() < 8 || &bytes[0..3] != b"CDF" {
                return Err(MpasError::Refusal(format!(
                    "{} is not a classic netCDF file (no CDF magic); the culler \
                     subsets classic bytes and cannot read HDF5",
                    path.display()
                )));
            }
            let version = bytes[3];
            let (nonneg8, offset8) = match version {
                1 => (false, false),
                2 => (false, true),
                5 => (true, true),
                other => {
                    return Err(MpasError::Refusal(format!(
                        "classic version byte {other} is not CDF-1/2/5"
                    )))
                }
            };
            let mut c = Cursor {
                b: &bytes,
                pos: 4,
                nonneg8,
                offset8,
            };
            let numrecs = c.nonneg()?;
            if !nonneg8 && numrecs == u32::MAX as u64 {
                return Err(MpasError::Refusal(
                    "parent numrecs is STREAMING (0xFFFFFFFF); the record count is \
                     unknowable without walking the tail and the native cull would \
                     read it the same wrong way".to_string(),
                ));
            }

            // dim_list
            let tag = c.u32()?;
            let ndims = c.nonneg()? as usize;
            let mut dims = Vec::with_capacity(ndims);
            if tag == 0x0A {
                for _ in 0..ndims {
                    let name = c.name()?;
                    let len = c.nonneg()? as usize;
                    dims.push(RawDim {
                        name,
                        len,
                        unlimited: len == 0,
                    });
                }
            } else if !(tag == 0 && ndims == 0) {
                return Err(MpasError::Refusal(format!(
                    "classic header expected NC_DIMENSION, found tag 0x{tag:X}"
                )));
            }

            let gattrs = c.attr_list()?;

            // var_list
            let tag = c.u32()?;
            let nvars = c.nonneg()? as usize;
            let mut vars: Vec<RawVar> = Vec::with_capacity(nvars);
            if tag == 0x0B {
                for _ in 0..nvars {
                    let name = c.name()?;
                    let nd = c.nonneg()? as usize;
                    let mut dimids = Vec::with_capacity(nd);
                    for _ in 0..nd {
                        dimids.push(c.nonneg()? as usize);
                    }
                    let attrs = c.attr_list()?;
                    let nc_type = c.u32()?;
                    let _vsize = c.nonneg()?;
                    let begin = c.offset()?;
                    for &d in &dimids {
                        if d >= dims.len() {
                            return Err(MpasError::Refusal(format!(
                                "variable {name} names dimid {d} but only {} dims exist",
                                dims.len()
                            )));
                        }
                    }
                    let is_record = dimids.first().is_some_and(|&d| dims[d].unlimited);
                    let mut elems: usize = 1;
                    for (axis, &d) in dimids.iter().enumerate() {
                        if is_record && axis == 0 {
                            continue;
                        }
                        if dims[d].unlimited {
                            return Err(MpasError::Refusal(format!(
                                "variable {name} uses the record dimension past axis 0"
                            )));
                        }
                        elems = elems.checked_mul(dims[d].len).ok_or_else(|| {
                            MpasError::Refusal(format!("variable {name} slab overflows usize"))
                        })?;
                    }
                    let width = type_size(nc_type)?;
                    let slab_bytes = elems * width;
                    let slab_stride = ((slab_bytes as u64) + 3) & !3;
                    vars.push(RawVar {
                        name,
                        dimids,
                        attrs,
                        nc_type,
                        begin,
                        is_record,
                        slab_bytes,
                        slab_stride,
                    });
                }
            } else if !(tag == 0 && nvars == 0) {
                return Err(MpasError::Refusal(format!(
                    "classic header expected NC_VARIABLE, found tag 0x{tag:X}"
                )));
            }

            // Record stride, honouring the sole-record-variable exception.
            let record_vars: Vec<usize> = (0..vars.len()).filter(|&i| vars[i].is_record).collect();
            if record_vars.len() == 1 {
                let i = record_vars[0];
                if matches!(vars[i].nc_type, NC_CHAR | NC_BYTE | NC_SHORT) {
                    vars[i].slab_stride = vars[i].slab_bytes as u64;
                }
            }
            let record_stride: u64 = record_vars.iter().map(|&i| vars[i].slab_stride).sum();

            Ok(RawNc {
                bytes,
                version,
                numrecs,
                dims,
                gattrs,
                vars,
                record_stride,
            })
        }

        pub fn var(&self, name: &str) -> Option<&RawVar> {
            self.vars.iter().find(|v| v.name == name)
        }

        pub fn dim_len(&self, name: &str) -> MpasResult<usize> {
            self.dims
                .iter()
                .find(|d| d.name == name)
                .map(|d| if d.unlimited { self.numrecs as usize } else { d.len })
                .ok_or_else(|| {
                    MpasError::Refusal(format!("parent has no {name} dimension"))
                })
        }

        /// One slab's raw bytes (whole array for fixed vars, one record for
        /// record vars).
        pub fn slab(&self, v: &RawVar, record: u64) -> MpasResult<&[u8]> {
            let start = if v.is_record {
                v.begin + record * self.record_stride
            } else {
                if record != 0 {
                    return Err(MpasError::Refusal(format!(
                        "{} is a fixed variable; record {record} does not exist",
                        v.name
                    )));
                }
                v.begin
            } as usize;
            let end = start + v.slab_bytes;
            if end > self.bytes.len() {
                return Err(MpasError::Refusal(format!(
                    "variable {} slab runs past the end of the file ({} > {})",
                    v.name,
                    end,
                    self.bytes.len()
                )));
            }
            Ok(&self.bytes[start..end])
        }

        pub fn read_i32(&self, name: &str) -> MpasResult<Vec<i32>> {
            let v = self
                .var(name)
                .ok_or_else(|| MpasError::Refusal(format!("parent has no variable {name}")))?;
            if v.nc_type != NC_INT {
                return Err(MpasError::Refusal(format!(
                    "parent {name} is nc_type {} where NC_INT was required",
                    v.nc_type
                )));
            }
            if v.is_record {
                return Err(MpasError::Refusal(format!(
                    "parent {name} is a record variable; the mesh topology the cull \
                     walks must be time-invariant"
                )));
            }
            Ok(self
                .slab(v, 0)?
                .chunks_exact(4)
                .map(|c| i32::from_be_bytes(c.try_into().unwrap()))
                .collect())
        }

        /// Cell coordinates as f64. A grid parent stores NC_DOUBLE; a static
        /// parent stores NC_FLOAT, which the native tool's numpy reader
        /// promotes exactly -- and the pinned records show the static cull
        /// selecting the identical element set as its grid cull, so the
        /// promoted walk reproduces both.
        pub fn read_coords(&self, name: &str) -> MpasResult<Vec<f64>> {
            let v = self
                .var(name)
                .ok_or_else(|| MpasError::Refusal(format!("parent has no variable {name}")))?;
            if v.is_record {
                return Err(MpasError::Refusal(format!(
                    "parent {name} is a record variable; cell coordinates must be \
                     time-invariant"
                )));
            }
            match v.nc_type {
                NC_DOUBLE => Ok(self
                    .slab(v, 0)?
                    .chunks_exact(8)
                    .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
                    .collect()),
                NC_FLOAT => Ok(self
                    .slab(v, 0)?
                    .chunks_exact(4)
                    .map(|c| f32::from_be_bytes(c.try_into().unwrap()) as f64)
                    .collect()),
                other => Err(MpasError::Refusal(format!(
                    "parent {name} is nc_type {other} where a float coordinate was \
                     required; the boundary walk has no defined arithmetic there"
                ))),
            }
        }
    }
}

use raw::{RawAttr, RawNc, RawVar};

// ===========================================================================
// Spherical helpers, transcribed from the native tool
// ===========================================================================
//
// These deliberately mirror limited_area/mesh.py operation-for-operation
// rather than reusing `geom.rs`: the walk's SELECTIONS (nearest cell, next
// path cell) are comparisons of transcendental expressions, and the safest
// route to the same selections is the same expression shapes.

#[inline]
fn latlon_to_xyz(lat: f64, lon: f64) -> [f64; 3] {
    [
        lon.cos() * lat.cos(),
        lon.sin() * lat.cos(),
        lat.sin(),
    ]
}

/// Haversine arc length, the tool's `sphere_distance` with radius 1.
#[inline]
fn sphere_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let s1 = (0.5 * (lat2 - lat1)).sin();
    let s2 = (0.5 * (lon2 - lon1)).sin();
    2.0 * (s1 * s1 + lat1.cos() * lat2.cos() * s2 * s2).sqrt().asin()
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// The tool's `rotate_about_vector` (Glenn Murray's formulas), verbatim.
/// Also MPAS's `mpas_rotate_about_vector` with the rotation origin at 0 --
/// the static builder's `mpas_in_cell` port reflects through it.
pub(crate) fn rotate_about_vector(x: [f64; 3], u: [f64; 3], theta: f64) -> [f64; 3] {
    let (xx, y, z) = (x[0], x[1], x[2]);
    let (a, b, w) = (u[0], u[1], u[2]);
    let vw2 = b * b + w * w;
    let uw2 = a * a + w * w;
    let uv2 = a * a + b * b;
    let m = (a * a + b * b + w * w).sqrt();
    let ct = theta.cos();
    let st = theta.sin();
    [
        (a * (a * xx + b * y + w * z) + (xx * vw2 + a * (-b * y - w * z)) * ct + m * (-w * y + b * z) * st) / (m * m),
        (b * (a * xx + b * y + w * z) + (y * uw2 + b * (-a * xx - w * z)) * ct + m * (w * xx - a * z) * st) / (m * m),
        (w * (a * xx + b * y + w * z) + (z * uv2 + w * (-a * xx - b * y)) * ct + m * (-b * xx + a * y) * st) / (m * m),
    ]
}

/// The tool's `xyz_to_latlon`, verbatim including its epsilon ladder.
fn xyz_to_latlon(p: [f64; 3]) -> (f64, f64) {
    let (x, y, z) = (p[0], p[1], p[2]);
    let eps = 1.0e-10f64;
    let lat = z.asin();
    let lon;
    if x.abs() > eps {
        if y.abs() > eps {
            let mut l = (y / x).abs().atan();
            if x <= 0.0 && y >= 0.0 {
                l = std::f64::consts::PI - l;
            } else if x <= 0.0 && y <= 0.0 {
                l += std::f64::consts::PI;
            } else if x >= 0.0 && y <= 0.0 {
                l = 2.0 * std::f64::consts::PI - l;
            }
            lon = l;
        } else if x > 0.0 {
            lon = 0.0;
        } else {
            lon = std::f64::consts::PI;
        }
    } else if y.abs() > eps {
        if y > 0.0 {
            lon = 0.5 * std::f64::consts::PI;
        } else {
            lon = 1.5 * std::f64::consts::PI;
        }
    } else {
        lon = 0.0;
    }
    (lat, lon)
}

// ===========================================================================
// The cell topology the mark phase walks
// ===========================================================================

/// The slice of the global mesh the cell-mark phase needs. Borrowed so tests
/// can drive the phase from a golden container without a netCDF file.
pub struct CellTopo<'a> {
    pub n_cells: usize,
    pub max_edges: usize,
    /// Radians.
    pub lat: &'a [f64],
    /// Radians, any wrap.
    pub lon: &'a [f64],
    pub n_edges_on_cell: &'a [i32],
    /// 1-based, row-major `[n_cells * max_edges]`, exactly as the file stores it.
    pub cells_on_cell: &'a [i32],
}

impl CellTopo<'_> {
    fn check(&self) -> MpasResult<()> {
        if self.lat.len() != self.n_cells
            || self.lon.len() != self.n_cells
            || self.n_edges_on_cell.len() != self.n_cells
            || self.cells_on_cell.len() != self.n_cells * self.max_edges
        {
            return Err(MpasError::Refusal(format!(
                "cell topology shapes disagree with nCells={}: lat {}, lon {}, \
                 nEdgesOnCell {}, cellsOnCell {}; a walk over misaligned arrays \
                 marks the wrong cells silently",
                self.n_cells,
                self.lat.len(),
                self.lon.len(),
                self.n_edges_on_cell.len(),
                self.cells_on_cell.len()
            )));
        }
        for c in 0..self.n_cells {
            let deg = self.n_edges_on_cell[c];
            if deg < 1 || deg as usize > self.max_edges {
                return Err(MpasError::Refusal(format!(
                    "cell {c} has nEdgesOnCell={deg} outside 1..={}; its neighbour \
                     row cannot be walked",
                    self.max_edges
                )));
            }
            for s in 0..deg as usize {
                let v = self.cells_on_cell[c * self.max_edges + s];
                if v < 1 || v as usize > self.n_cells {
                    return Err(MpasError::Refusal(format!(
                        "parent cellsOnCell[cell={c},slot={s}]={v} is outside \
                         1..={} in a VALID slot; the native tool would index the \
                         mask array out of range there, so this parent cannot be \
                         culled compatibly",
                        self.n_cells
                    )));
                }
            }
        }
        Ok(())
    }

    /// Greedy nearest-cell descent, the tool's `MeshHandler.nearest_cell`:
    /// start at cell 0, move to any neighbour at least as close, repeat until
    /// stationary.
    pub fn nearest_cell(&self, lat: f64, lon: f64) -> usize {
        let mut nearest = 0usize;
        let mut current = usize::MAX;
        while nearest != current {
            current = nearest;
            let current_distance =
                sphere_distance(self.lat[current], self.lon[current], lat, lon);
            nearest = current;
            let mut nearest_distance = current_distance;
            for s in 0..self.n_edges_on_cell[current] as usize {
                let i = (self.cells_on_cell[current * self.max_edges + s] - 1) as usize;
                let d = sphere_distance(self.lat[i], self.lon[i], lat, lon);
                if d <= nearest_distance {
                    nearest = i;
                    nearest_distance = d;
                }
            }
        }
        nearest
    }
}

/// What the mark phase measured, for the receipt.
#[derive(Debug, Clone, Serialize)]
pub struct MarkStats {
    /// Parent index (0-based) of the flood-fill seed cell.
    pub seed_cell_index0: usize,
    /// Cells marked by the boundary walk before the fill.
    pub boundary_path_cells: usize,
    /// Cells at each output mask value 0..7, in order.
    pub ring_cell_counts: [usize; 8],
}

/// Mark the region on the GLOBAL mesh: boundary walk, flood fill, then the
/// eight native layers. Returns the global mask with the tool's INTERNAL
/// values: 0 outside, 1 interior, 2..8 rings outward.
pub fn mark_region_cells(
    topo: &CellTopo<'_>,
    boundary_loops: &[Vec<(f64, f64)>],
    seed: (f64, f64),
) -> MpasResult<(Vec<i32>, MarkStats)> {
    topo.check()?;
    let mut mask = vec![UNMARKED; topo.n_cells];

    // --- boundary walk, one loop at a time --------------------------------
    for points in boundary_loops {
        mark_boundary(topo, points, &mut mask)?;
    }
    let boundary_path_cells = mask.iter().filter(|&&m| m == INSIDE).count();

    // --- flood fill from the seed's nearest cell --------------------------
    let in_cell = topo.nearest_cell(seed.0, seed.1);
    if mask[in_cell] == INSIDE {
        return Err(MpasError::Refusal(format!(
            "the interior seed's nearest cell {in_cell} is ON the boundary path; \
             a flood fill from a path cell leaks to both sides and marks the \
             whole sphere. Pick a region whose interior point sits inside the \
             walked boundary"
        )));
    }
    {
        let mut stack = vec![in_cell];
        while let Some(ic) = stack.pop() {
            for s in 0..topo.n_edges_on_cell[ic] as usize {
                let j = (topo.cells_on_cell[ic * topo.max_edges + s] - 1) as usize;
                if mask[j] == UNMARKED {
                    mask[j] = INSIDE;
                    stack.push(j);
                }
            }
        }
        // The seed itself is marked by its own neighbours' sweeps in the
        // native tool; a 1-cell region would leave it unmarked, so state it.
        if mask[in_cell] == UNMARKED {
            return Err(MpasError::Refusal(format!(
                "flood fill marked nothing around seed cell {in_cell}; the region \
                 is smaller than one cell and has no interior to keep"
            )));
        }
    }

    // --- the eight native layers ------------------------------------------
    // Layer L walks cells with mask in [1, L) from the seed and gives every
    // unmarked neighbour the value L. Transcribed with the tool's own
    // negate-while-visiting trick so the visit set is identical.
    for layer in 1..=NUM_BOUNDARY_LAYERS {
        let mut stack = vec![in_cell];
        while let Some(ic) = stack.pop() {
            for s in 0..topo.n_edges_on_cell[ic] as usize {
                let j = (topo.cells_on_cell[ic * topo.max_edges + s] - 1) as usize;
                if mask[j] >= INSIDE && layer > mask[j] {
                    mask[j] = -mask[j];
                    stack.push(j);
                } else if mask[j] == 0 {
                    mask[j] = layer;
                }
            }
        }
        for m in mask.iter_mut() {
            *m = m.abs();
        }
    }

    let mut ring_cell_counts = [0usize; 8];
    for &m in &mask {
        if m > 0 {
            ring_cell_counts[(m - 1) as usize] += 1;
        }
    }

    // The measured native invariant: adjacent marked cells never differ by
    // more than 1. A violation means the layer growth above diverged.
    for c in 0..topo.n_cells {
        if mask[c] == 0 {
            continue;
        }
        for s in 0..topo.n_edges_on_cell[c] as usize {
            let j = (topo.cells_on_cell[c * topo.max_edges + s] - 1) as usize;
            if mask[j] != 0 && (mask[c] - mask[j]).abs() > 1 {
                return Err(MpasError::Refusal(format!(
                    "cells {c} and {j} carry masks {} and {} yet are neighbours; \
                     the native rings never jump by more than 1, so this mask \
                     would not byte-match any native cull",
                    mask[c], mask[j]
                )));
            }
        }
    }

    Ok((
        mask,
        MarkStats {
            seed_cell_index0: in_cell,
            boundary_path_cells,
            ring_cell_counts,
        },
    ))
}

/// The tool's `mark_boundary`: nearest cells to the loop points, connected
/// pairwise by a great-circle cell walk.
fn mark_boundary(
    topo: &CellTopo<'_>,
    points: &[(f64, f64)],
    mask: &mut [i32],
) -> MpasResult<()> {
    if points.len() < 2 {
        return Err(MpasError::Refusal(format!(
            "a boundary loop of {} point(s) encloses nothing",
            points.len()
        )));
    }
    let boundary_cells: Vec<usize> = points
        .iter()
        .map(|&(lat, lon)| topo.nearest_cell(lat, lon))
        .collect();
    for &c in &boundary_cells {
        mask[c] = INSIDE;
    }

    for i in 0..boundary_cells.len() {
        let source = boundary_cells[i];
        let target = boundary_cells[(i + 1) % boundary_cells.len()];
        if source == target {
            continue;
        }
        let pta = latlon_to_xyz(topo.lat[source], topo.lon[source]);
        let ptb = latlon_to_xyz(topo.lat[target], topo.lon[target]);
        let mut plane = cross3(pta, ptb);
        let n = norm3(plane);
        plane = [plane[0] / n, plane[1] / n, plane[2] / n];

        let mut i_cell = source;
        // The native walk keeps `k` across iterations; a step where no
        // neighbour qualifies would silently reuse the previous k there. Here
        // it is a named refusal instead, because that reuse loops forever.
        let mut steps = 0usize;
        while i_cell != target {
            mask[i_cell] = INSIDE;
            let mut minangle = f64::INFINITY;
            let mut k: Option<usize> = None;
            let mindist = sphere_distance(
                topo.lat[i_cell],
                topo.lon[i_cell],
                topo.lat[target],
                topo.lon[target],
            );
            for s in 0..topo.n_edges_on_cell[i_cell] as usize {
                let v = (topo.cells_on_cell[i_cell * topo.max_edges + s] - 1) as usize;
                let dist = sphere_distance(
                    topo.lat[v],
                    topo.lon[v],
                    topo.lat[target],
                    topo.lon[target],
                );
                if dist > mindist {
                    continue;
                }
                let pt = latlon_to_xyz(topo.lat[v], topo.lon[v]);
                let angle = (0.5 * std::f64::consts::PI - dot3(plane, pt).acos()).abs();
                if angle < minangle {
                    minangle = angle;
                    k = Some(v);
                }
            }
            i_cell = k.ok_or_else(|| {
                MpasError::Refusal(format!(
                    "boundary walk stalled at cell {i_cell}: no neighbour is at \
                     least as close to the target as the cell itself. The native \
                     tool loops forever here; the region's boundary segment \
                     cannot be traced on this mesh"
                ))
            })?;
            steps += 1;
            if steps > topo.n_cells {
                return Err(MpasError::Refusal(
                    "boundary walk exceeded nCells steps without reaching its \
                     target; the segment is cycling and the native tool would \
                     never terminate either"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

/// Edge masks from cell masks on the GLOBAL mesh: MIN of the two adjacent
/// cells when both are marked, else MAX.
pub fn edge_masks_from_cells(
    cell_mask: &[i32],
    cells_on_edge: &[i32],
    n_edges: usize,
) -> MpasResult<Vec<i32>> {
    if cells_on_edge.len() != n_edges * 2 {
        return Err(MpasError::Refusal(format!(
            "cellsOnEdge has {} values where nEdges={n_edges} demands {}",
            cells_on_edge.len(),
            n_edges * 2
        )));
    }
    let mut out = vec![0i32; n_edges];
    for e in 0..n_edges {
        let mut vals = [0i32; 2];
        for s in 0..2 {
            let v = cells_on_edge[e * 2 + s];
            if v < 1 || v as usize > cell_mask.len() {
                return Err(MpasError::Refusal(format!(
                    "parent cellsOnEdge[edge={e},slot={s}]={v} is outside 1..={}; \
                     the native mask rule reads the LAST cell's mask for such a \
                     slot, which is not a rule at all, so this parent is refused",
                    cell_mask.len()
                )));
            }
            vals[s] = cell_mask[(v - 1) as usize];
        }
        let mn = vals[0].min(vals[1]);
        out[e] = if mn > 0 { mn } else { vals[0].max(vals[1]) };
    }
    Ok(out)
}

/// Vertex masks from cell masks on the GLOBAL mesh, same MIN-else-MAX rule
/// over the `vertexDegree` adjacent cells.
pub fn vertex_masks_from_cells(
    cell_mask: &[i32],
    cells_on_vertex: &[i32],
    n_vertices: usize,
    vertex_degree: usize,
) -> MpasResult<Vec<i32>> {
    if cells_on_vertex.len() != n_vertices * vertex_degree {
        return Err(MpasError::Refusal(format!(
            "cellsOnVertex has {} values where nVertices={n_vertices} x degree \
             {vertex_degree} demands {}",
            cells_on_vertex.len(),
            n_vertices * vertex_degree
        )));
    }
    let mut out = vec![0i32; n_vertices];
    for vx in 0..n_vertices {
        let mut mn = i32::MAX;
        let mut mx = i32::MIN;
        for s in 0..vertex_degree {
            let v = cells_on_vertex[vx * vertex_degree + s];
            if v < 1 || v as usize > cell_mask.len() {
                return Err(MpasError::Refusal(format!(
                    "parent cellsOnVertex[vertex={vx},slot={s}]={v} is outside \
                     1..={}; the native mask rule reads the LAST cell's mask for \
                     such a slot, which is not a rule at all, so this parent is \
                     refused",
                    cell_mask.len()
                )));
            }
            let m = cell_mask[(v - 1) as usize];
            mn = mn.min(m);
            mx = mx.max(m);
        }
        out[vx] = if mn > 0 { mn } else { mx };
    }
    Ok(out)
}

/// The tool's `scan()`: a global -> regional renumber map. Marked elements
/// get 1..K in parent order; unmarked stay 0.
pub fn renumber_map(mask: &[i32]) -> Vec<i32> {
    let mut out = vec![0i32; mask.len()];
    let mut next = 1i32;
    for (i, &m) in mask.iter().enumerate() {
        if m > 0 {
            out[i] = next;
            next += 1;
        }
    }
    out
}

// ===========================================================================
// Boundary points from a Shape row
// ===========================================================================

/// Turn a [`Shape`] row into the boundary loops and interior seed the mark
/// phase consumes. Everything in radians.
///
/// * `polygon` -- the vertices verbatim, joined by great-circle walks, which
///   is exactly the native tool's `custom` type. The seed is the normalised
///   vertex centroid ([`Shape::interior_point`]); any interior point selects
///   the same flood-fill component, so the output does not depend on it.
/// * `cap` -- the native `circle` type: 100 points swept around the centre by
///   `rotate_about_vector`, endpoints duplicated, same K-axis selection rule.
/// * `lat_lon_box` -- boundary sampled along its parallels and meridians at
///   `BOX_SAMPLES_PER_SIDE` points per side. The native tool has no box type;
///   a box's east-west sides follow PARALLELS here, where a 4-vertex polygon
///   would cut great circles, so the two are different regions by design.
pub fn boundary_from_shape(shape: &Shape) -> MpasResult<(Vec<Vec<(f64, f64)>>, (f64, f64))> {
    const BOX_SAMPLES_PER_SIDE: usize = 100;
    let loops: Vec<Vec<(f64, f64)>> = match shape {
        Shape::Polygon { vertices_deg } => {
            if vertices_deg.len() < 3 {
                return Err(MpasError::Refusal(format!(
                    "a polygon region of {} vertices encloses nothing",
                    vertices_deg.len()
                )));
            }
            vec![vertices_deg
                .iter()
                .map(|v| (v[0].to_radians(), v[1].to_radians()))
                .collect()]
        }
        Shape::Cap {
            center_deg,
            radius_km,
        } => {
            let lat = center_deg[0].to_radians();
            let lon = center_deg[1].to_radians();
            let radius = radius_km * 1000.0 / crate::mesh::geom::EARTH_RADIUS_M;
            if !(radius > 0.0 && radius < std::f64::consts::PI) {
                return Err(MpasError::Refusal(format!(
                    "cap radius {radius_km} km is {radius:.4} rad on the sphere; \
                     outside (0, pi) there is no circle to trace"
                )));
            }
            let c = latlon_to_xyz(lat, lon);
            // The native circle: K = x-hat unless the centre is near it.
            let mut k = [1.0, 0.0, 0.0];
            if dot3(k, c).abs() >= 0.9 {
                k = [0.0, 1.0, 0.0];
            }
            let mut s = cross3(c, k);
            let n = norm3(s);
            s = [s[0] / n, s[1] / n, s[2] / n];
            let p0 = rotate_about_vector(c, s, radius);
            // numpy linspace(0, 2*pi, 100): step computed once, endpoint set
            // exactly, mirrored so a future native circle oracle can match.
            let step = (2.0 * std::f64::consts::PI) / 99.0;
            let mut pts = Vec::with_capacity(100);
            for i in 0..100 {
                let r = if i == 99 {
                    2.0 * std::f64::consts::PI
                } else {
                    i as f64 * step
                };
                pts.push(xyz_to_latlon(rotate_about_vector(p0, c, r)));
            }
            vec![pts]
        }
        Shape::LatLonBox { lat_deg, lon_deg } => {
            let (lat0, lat1) = (
                lat_deg[0].to_radians().min(lat_deg[1].to_radians()),
                lat_deg[0].to_radians().max(lat_deg[1].to_radians()),
            );
            let lon0 = lon_deg[0].to_radians();
            let mut span = lon_deg[1].to_radians() - lon0;
            let tau = std::f64::consts::TAU;
            span = ((span % tau) + tau) % tau;
            if span == 0.0 {
                return Err(MpasError::Refusal(
                    "lat_lon_box longitude span is zero; the box has no width".to_string(),
                ));
            }
            let n = BOX_SAMPLES_PER_SIDE;
            let mut pts: Vec<(f64, f64)> = Vec::with_capacity(4 * n);
            // Counter-clockwise: south side west->east, east meridian up,
            // north side east->west, west meridian down.
            for i in 0..n {
                pts.push((lat0, lon0 + span * i as f64 / n as f64));
            }
            for i in 0..n {
                pts.push((lat0 + (lat1 - lat0) * i as f64 / n as f64, lon0 + span));
            }
            for i in 0..n {
                pts.push((lat1, lon0 + span * (n - i) as f64 / n as f64));
            }
            for i in 0..n {
                pts.push((lat1 - (lat1 - lat0) * i as f64 / n as f64, lon0));
            }
            vec![pts]
        }
    };
    let seed_xyz = shape.interior_point();
    let (lat, lon) = crate::mesh::geom::lat_lon(seed_xyz);
    Ok((loops, (lat, lon)))
}

// ===========================================================================
// The file-level cull
// ===========================================================================

/// Everything a cull measured, for the receipt file.
#[derive(Debug, Clone, Serialize)]
pub struct CullReceipt {
    pub engine: String,
    pub parent_file: String,
    pub parent_sha256: String,
    pub region: Shape,
    pub parent_cells: usize,
    pub parent_edges: usize,
    pub parent_vertices: usize,
    pub region_cells: usize,
    pub region_edges: usize,
    pub region_vertices: usize,
    /// V - E + F. A disk is 1; anything else is a torn or doubled region.
    pub euler_v_minus_e_plus_f: i64,
    pub mark: MarkStats,
    /// Edge and vertex counts at each output mask value 0..7.
    pub ring_edge_counts: [usize; 8],
    pub ring_vertex_counts: [usize; 8],
    /// True when the parent's LAST cell/edge/vertex is inside the region --
    /// the one condition under which the native tool's `map[field-1]` numpy
    /// wrap writes a real index where this culler writes the 0 sentinel.
    /// False on both pinned parents, so false means byte-identical.
    pub native_wrap_divergence: [bool; 3],
    pub output_file: String,
    pub output_bytes: u64,
    pub output_sha256: String,
    pub graph_file: Option<String>,
    pub graph_sha256: Option<String>,
}

/// Cull `parent` (grid or static, classic netCDF) to the region `shape`,
/// writing the regional file and, when asked, the METIS `graph.info`.
pub fn cull_file(
    parent_path: &Path,
    shape: &Shape,
    out_path: &Path,
    graph_path: Option<&Path>,
) -> MpasResult<CullReceipt> {
    let parent = RawNc::open(parent_path)?;

    let n_cells = parent.dim_len("nCells")?;
    let n_edges = parent.dim_len("nEdges")?;
    let n_vertices = parent.dim_len("nVertices")?;
    let max_edges = parent.dim_len("maxEdges")?;
    let vertex_degree = parent.dim_len("vertexDegree")?;

    // Refuse attribute conventions the native tool's netCDF4 reader would
    // reinterpret while this reader copies raw bytes.
    for v in &parent.vars {
        for a in &v.attrs {
            if matches!(a.name.as_str(), "scale_factor" | "add_offset") {
                return Err(MpasError::Refusal(format!(
                    "parent variable {} carries {}; the native cull decodes and \
                     re-encodes through that attribute while this culler copies \
                     raw bytes, and the two files would disagree on every value",
                    v.name, a.name
                )));
            }
        }
    }

    let lat_cell = parent.read_coords("latCell")?;
    let lon_cell = parent.read_coords("lonCell")?;
    let n_edges_on_cell = parent.read_i32("nEdgesOnCell")?;
    let cells_on_cell = parent.read_i32("cellsOnCell")?;
    let cells_on_edge = parent.read_i32("cellsOnEdge")?;
    let cells_on_vertex = parent.read_i32("cellsOnVertex")?;

    let topo = CellTopo {
        n_cells,
        max_edges,
        lat: &lat_cell,
        lon: &lon_cell,
        n_edges_on_cell: &n_edges_on_cell,
        cells_on_cell: &cells_on_cell,
    };

    let (loops, seed) = boundary_from_shape(shape)?;
    let (cell_mask, mark) = mark_region_cells(&topo, &loops, seed)?;

    if cell_mask.iter().filter(|&&m| m != 0).count() == n_cells {
        return Err(MpasError::Refusal(format!(
            "every one of the parent's {n_cells} cells is inside the region; a \
             cull that keeps the whole sphere is a mis-specified region, and \
             the native tool refuses it too"
        )));
    }

    let edge_mask = edge_masks_from_cells(&cell_mask, &cells_on_edge, n_edges)?;
    let vertex_mask = vertex_masks_from_cells(&cell_mask, &cells_on_vertex, n_vertices, vertex_degree)?;

    // Parent-order subset positions. The native tool selects rows through
    // indexTo*ID VALUES; on the (universal) identity numbering these are the
    // mask positions. A permuted parent numbering is refused rather than
    // silently subsetting different rows than the masks describe.
    for name in ["indexToCellID", "indexToEdgeID", "indexToVertexID"] {
        if let Some(v) = parent.var(name) {
            let ids = parent.read_i32(name)?;
            if ids.iter().enumerate().any(|(i, &id)| id != i as i32 + 1) {
                return Err(MpasError::Refusal(format!(
                    "parent {} is not the identity numbering; the native tool \
                     subsets rows through these VALUES while the masks are \
                     positional, and on a permuted parent the two select \
                     different elements",
                    v.name
                )));
            }
        }
    }

    let cell_ids: Vec<usize> = (0..n_cells).filter(|&i| cell_mask[i] != 0).collect();
    let edge_ids: Vec<usize> = (0..n_edges).filter(|&i| edge_mask[i] != 0).collect();
    let vertex_ids: Vec<usize> = (0..n_vertices).filter(|&i| vertex_mask[i] != 0).collect();

    let cell_map = renumber_map(&cell_mask);
    let edge_map = renumber_map(&edge_mask);
    let vertex_map = renumber_map(&vertex_mask);

    let native_wrap_divergence = [
        cell_mask[n_cells - 1] != 0,
        edge_mask[n_edges - 1] != 0,
        vertex_mask[n_vertices - 1] != 0,
    ];

    // ---- output schema ----------------------------------------------------
    let format = match parent.version {
        1 => NcFormat::Classic,
        2 => NcFormat::Offset64,
        5 => NcFormat::Data64,
        _ => unreachable!("RawNc::open admits only 1/2/5"),
    };

    let mut dims: Vec<NcDim> = Vec::with_capacity(parent.dims.len());
    let mut dimid_of_cells = None;
    let mut dimid_of_edges = None;
    let mut dimid_of_vertices = None;
    for (i, d) in parent.dims.iter().enumerate() {
        if d.unlimited {
            dims.push(NcDim::record(&d.name));
            continue;
        }
        let len = match d.name.as_str() {
            "nCells" => {
                dimid_of_cells = Some(i);
                cell_ids.len()
            }
            "nEdges" => {
                dimid_of_edges = Some(i);
                edge_ids.len()
            }
            "nVertices" => {
                dimid_of_vertices = Some(i);
                vertex_ids.len()
            }
            _ => d.len,
        };
        dims.push(NcDim::fixed(&d.name, len));
    }
    let d_cells = dimid_of_cells
        .ok_or_else(|| MpasError::Refusal("parent nCells is the record dimension".to_string()))?;
    let d_edges = dimid_of_edges
        .ok_or_else(|| MpasError::Refusal("parent nEdges is the record dimension".to_string()))?;
    let d_vertices = dimid_of_vertices
        .ok_or_else(|| MpasError::Refusal("parent nVertices is the record dimension".to_string()))?;

    // Global attributes: on_a_sphere then sphere_radius, parent bytes.
    let mut gattrs: Vec<NcAttr> = Vec::with_capacity(2);
    for name in ["on_a_sphere", "sphere_radius"] {
        let a = parent
            .gattrs
            .iter()
            .find(|a| a.name == name)
            .ok_or_else(|| {
                MpasError::Refusal(format!(
                    "parent has no global attribute {name}; the native cull dies \
                     on the same absence, so the region file cannot declare its \
                     sphere"
                ))
            })?;
        gattrs.push(NcAttr {
            name: a.name.clone(),
            value: attr_value(a)?,
        });
    }

    // THE COORDINATE FRAME CROSSES THE CULL, OR THE CHILD IS AMBIGUOUS.
    //
    // A cull SUBSETS its parent: every coordinate it writes is a parent byte,
    // at the parent's own width, so the child is in the parent's frame by
    // construction and inherits its declaration rather than minting one.  The
    // native MPAS-Limited-Area cull copies exactly two global attributes and
    // this reproduces that byte for byte -- a NATIVE parent declares nothing,
    // so nothing is added and the child's bytes are unchanged.  A parent this
    // generator made declares `rw_coordinate_representation`, and dropping it
    // here would hand a reader binary64 arrays under the binary32 default: a
    // storage tolerance 5.4e8 times too loose, which admits a dvEdge corrupted
    // by a metre.  The declaration is checked against the parent's actual
    // dtype first, so a parent whose attribute and arrays disagree is refused
    // by name here rather than carried forward.
    {
        use crate::staticfile::coordframe;
        let parent_says = parent
            .gattrs
            .iter()
            .find(|a| a.name == coordframe::REPRESENTATION_ATTR)
            .map(|a| match attr_value(a)? {
                rw_store::netcdf_classic::NcAttrValue::Text(t) => Ok(t),
                other => Err(MpasError::Refusal(format!(
                    "parent {} is {other:?}, not text; a frame that cannot be read is a frame                      that would be guessed",
                    coordframe::REPRESENTATION_ATTR
                ))),
            })
            .transpose()?;
        let stored_is_binary64 = parent
            .vars
            .iter()
            .find(|v| v.name == "xCell")
            .map(|v| v.nc_type == raw::NC_DOUBLE)
            .unwrap_or(false);
        let radius = parent
            .gattrs
            .iter()
            .find(|a| a.name == "sphere_radius")
            .and_then(|a| attr_value(a).ok())
            .and_then(|v| match v {
                rw_store::netcdf_classic::NcAttrValue::Doubles(d) => d.first().copied(),
                rw_store::netcdf_classic::NcAttrValue::Floats(f) => f.first().map(|&x| x as f64),
                _ => None,
            })
            .unwrap_or(crate::mesh::geom::EARTH_RADIUS_M);
        let rep = coordframe::verify_declaration(
            "cull parent",
            parent_says.as_deref(),
            stored_is_binary64,
            None,
            radius,
        )?;
        if parent_says.is_some() {
            gattrs.extend(rep.declaration_attributes(radius));
        }
    }

    // Variables: the three masks first, then the parent's variables in parent
    // order, minus any parent bdyMask* (a static parent carries stale ones).
    let mut vars: Vec<NcVarDef> = Vec::with_capacity(parent.vars.len() + 3);
    vars.push(NcVarDef::new("bdyMaskCell", NcType::Int, vec![d_cells]));
    vars.push(NcVarDef::new("bdyMaskEdge", NcType::Int, vec![d_edges]));
    vars.push(NcVarDef::new("bdyMaskVertex", NcType::Int, vec![d_vertices]));

    let copied: Vec<&RawVar> = parent
        .vars
        .iter()
        .filter(|v| !matches!(v.name.as_str(), "bdyMaskCell" | "bdyMaskEdge" | "bdyMaskVertex"))
        .collect();
    for v in &copied {
        let ty = match v.nc_type {
            raw::NC_CHAR => NcType::Char,
            raw::NC_SHORT => NcType::Short,
            raw::NC_INT => NcType::Int,
            raw::NC_FLOAT => NcType::Float,
            raw::NC_DOUBLE => NcType::Double,
            other => {
                return Err(MpasError::Refusal(format!(
                    "parent variable {} is nc_type {other}, which the classic \
                     writer does not carry; the cull cannot reproduce its bytes",
                    v.name
                )))
            }
        };
        // The native attribute rule: try units, then long_name, abandon at
        // the first absence. A variable with long_name but no units gets
        // NEITHER -- that is the tool's measured behaviour, not a choice here.
        let mut attrs: Vec<NcAttr> = Vec::new();
        if let Some(u) = v.attrs.iter().find(|a| a.name == "units") {
            attrs.push(NcAttr {
                name: "units".to_string(),
                value: attr_value(u)?,
            });
            if let Some(l) = v.attrs.iter().find(|a| a.name == "long_name") {
                attrs.push(NcAttr {
                    name: "long_name".to_string(),
                    value: attr_value(l)?,
                });
            }
        }
        vars.push(NcVarDef::new(&v.name, ty, v.dimids.clone()).with_attrs(attrs));
    }

    if out_path.exists() {
        return Err(MpasError::Refusal(format!(
            "{} already exists; a cull never overwrites silently",
            out_path.display()
        )));
    }
    let mut w = NcClassicWriter::create(
        out_path,
        format,
        dims,
        gattrs,
        vars,
        parent.numrecs,
    )
    .map_err(|e| MpasError::Refusal(format!("regional writer: {e}")))?;

    // ---- data -------------------------------------------------------------
    let masked_out = |mask: &[i32], ids: &[usize]| -> Vec<i32> {
        ids.iter().map(|&i| mask[i] - 1).collect()
    };
    w.put("bdyMaskCell", NcData::Ints(&masked_out(&cell_mask, &cell_ids)))
        .map_err(wr)?;
    w.put("bdyMaskEdge", NcData::Ints(&masked_out(&edge_mask, &edge_ids)))
        .map_err(wr)?;
    w.put(
        "bdyMaskVertex",
        NcData::Ints(&masked_out(&vertex_mask, &vertex_ids)),
    )
    .map_err(wr)?;

    for v in &copied {
        // Which element family does this variable subset over? The native
        // tool tests for nCells first, then nEdges, then nVertices, at ANY
        // axis, and always slices the leading non-record axis.
        let dim_names: Vec<&str> = v.dimids.iter().map(|&d| parent.dims[d].name.as_str()).collect();
        let class = if dim_names.contains(&"nCells") {
            Some((ElemClass::Cell, &cell_ids))
        } else if dim_names.contains(&"nEdges") {
            Some((ElemClass::Edge, &edge_ids))
        } else if dim_names.contains(&"nVertices") {
            Some((ElemClass::Vertex, &vertex_ids))
        } else {
            None
        };

        let reindex_with = INDEXING_FIELDS
            .iter()
            .find(|(n, _)| *n == v.name)
            .map(|&(_, c)| match c {
                ElemClass::Cell => &cell_map,
                ElemClass::Edge => &edge_map,
                ElemClass::Vertex => &vertex_map,
            });

        let records = if v.is_record { parent.numrecs } else { 1 };
        for r in 0..records {
            let slab = parent.slab(v, r)?;
            let out_bytes: Vec<u8> = match (class, reindex_with) {
                (None, None) => slab.to_vec(),
                (None, Some(_)) => {
                    return Err(MpasError::Refusal(format!(
                        "variable {} is an indexing field with no element \
                         dimension; its values cannot be renumbered",
                        v.name
                    )))
                }
                (Some((cls, ids)), maybe_map) => {
                    // The subset axis is the leading non-record axis; the
                    // element dimension must BE that axis, or the native tool
                    // crashes on a shape mismatch and so does this refusal.
                    let lead_axis = if v.is_record { 1 } else { 0 };
                    let lead_dim = v.dimids.get(lead_axis).map(|&d| parent.dims[d].name.as_str());
                    let elem_dim = match cls {
                        ElemClass::Cell => "nCells",
                        ElemClass::Edge => "nEdges",
                        ElemClass::Vertex => "nVertices",
                    };
                    if lead_dim != Some(elem_dim) {
                        return Err(MpasError::Refusal(format!(
                            "variable {} carries {elem_dim} away from its leading \
                             axis ({:?}); the native tool slices the leading axis \
                             regardless and dies on the shape, so the layout is \
                             refused by name instead",
                            v.name, dim_names
                        )));
                    }
                    let n_lead = parent.dims[v.dimids[lead_axis]].len;
                    if n_lead == 0 {
                        return Err(MpasError::Refusal(format!(
                            "variable {} leading dimension is empty", v.name
                        )));
                    }
                    let row_bytes = v.slab_bytes / n_lead;
                    match maybe_map {
                        None => {
                            let mut out = Vec::with_capacity(ids.len() * row_bytes);
                            for &i in ids.iter() {
                                out.extend_from_slice(&slab[i * row_bytes..(i + 1) * row_bytes]);
                            }
                            out
                        }
                        Some(map) => {
                            if v.nc_type != raw::NC_INT {
                                return Err(MpasError::Refusal(format!(
                                    "indexing field {} is not NC_INT", v.name
                                )));
                            }
                            let row_vals = row_bytes / 4;
                            let mut out = Vec::with_capacity(ids.len() * row_bytes);
                            for &i in ids.iter() {
                                let row = &slab[i * row_bytes..(i + 1) * row_bytes];
                                for c in row.chunks_exact(4) {
                                    let old = i32::from_be_bytes(c.try_into().unwrap());
                                    let new = remap(old, map, &v.name)?;
                                    out.extend_from_slice(&new.to_be_bytes());
                                }
                            }
                            debug_assert_eq!(out.len(), ids.len() * row_vals * 4);
                            out
                        }
                    }
                }
            };
            put_raw(&mut w, v, r, &out_bytes)?;
        }
    }

    w.finish().map_err(wr)?;

    // ---- graph.info -------------------------------------------------------
    let mut graph_sha256 = None;
    if let Some(gp) = graph_path {
        write_graph_info(
            gp,
            &cell_ids,
            &edge_ids,
            &n_edges_on_cell,
            &cells_on_cell,
            &cells_on_edge,
            &cell_map,
            max_edges,
        )?;
        graph_sha256 = Some(crate::sha256_file(gp)?);
    }

    let region_cells = cell_ids.len();
    let region_edges = edge_ids.len();
    let region_vertices = vertex_ids.len();
    let mut ring_edge_counts = [0usize; 8];
    for &i in &edge_ids {
        ring_edge_counts[(edge_mask[i] - 1) as usize] += 1;
    }
    let mut ring_vertex_counts = [0usize; 8];
    for &i in &vertex_ids {
        ring_vertex_counts[(vertex_mask[i] - 1) as usize] += 1;
    }

    Ok(CullReceipt {
        engine: concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)").to_string(),
        parent_file: parent_path.display().to_string(),
        parent_sha256: crate::sha256_file(parent_path)?,
        region: shape.clone(),
        parent_cells: n_cells,
        parent_edges: n_edges,
        parent_vertices: n_vertices,
        region_cells,
        region_edges,
        region_vertices,
        euler_v_minus_e_plus_f: region_vertices as i64 - region_edges as i64 + region_cells as i64,
        mark,
        ring_edge_counts,
        ring_vertex_counts,
        native_wrap_divergence,
        output_file: out_path.display().to_string(),
        output_bytes: std::fs::metadata(out_path)?.len(),
        output_sha256: crate::sha256_file(out_path)?,
        graph_file: graph_path.map(|p| p.display().to_string()),
        graph_sha256,
    })
}

fn wr(e: rw_store::error::RwStoreError) -> MpasError {
    MpasError::Refusal(format!("regional writer: {e}"))
}

/// Reindex one stored value through a renumber map. Stored 0 stays 0 -- the
/// documented divergence from the native tool's numpy wrap; see the module
/// docs and [`CullReceipt::native_wrap_divergence`].
#[inline]
fn remap(old: i32, map: &[i32], name: &str) -> MpasResult<i32> {
    if old == 0 {
        return Ok(0);
    }
    if old < 0 || old as usize > map.len() {
        return Err(MpasError::Refusal(format!(
            "{name} carries index {old}, outside 1..={}; the parent's own \
             connectivity is broken and no cull of it can be right",
            map.len()
        )));
    }
    Ok(map[(old - 1) as usize])
}

/// Push one slab of raw big-endian bytes through the typed writer.
fn put_raw(
    w: &mut NcClassicWriter,
    v: &RawVar,
    record: u64,
    bytes: &[u8],
) -> MpasResult<()> {
    let res = match v.nc_type {
        raw::NC_CHAR => {
            let data = bytes.to_vec();
            if v.is_record {
                w.put_record(&v.name, record, NcData::Chars(&data))
            } else {
                w.put(&v.name, NcData::Chars(&data))
            }
        }
        raw::NC_SHORT => {
            let data: Vec<i16> = bytes
                .chunks_exact(2)
                .map(|c| i16::from_be_bytes(c.try_into().unwrap()))
                .collect();
            if v.is_record {
                w.put_record(&v.name, record, NcData::Shorts(&data))
            } else {
                w.put(&v.name, NcData::Shorts(&data))
            }
        }
        raw::NC_INT => {
            let data: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_be_bytes(c.try_into().unwrap()))
                .collect();
            if v.is_record {
                w.put_record(&v.name, record, NcData::Ints(&data))
            } else {
                w.put(&v.name, NcData::Ints(&data))
            }
        }
        raw::NC_FLOAT => {
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_be_bytes(c.try_into().unwrap()))
                .collect();
            if v.is_record {
                w.put_record(&v.name, record, NcData::Floats(&data))
            } else {
                w.put(&v.name, NcData::Floats(&data))
            }
        }
        raw::NC_DOUBLE => {
            let data: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
                .collect();
            if v.is_record {
                w.put_record(&v.name, record, NcData::Doubles(&data))
            } else {
                w.put(&v.name, NcData::Doubles(&data))
            }
        }
        other => {
            return Err(MpasError::Refusal(format!(
                "variable {} nc_type {other} cannot be written", v.name
            )))
        }
    };
    res.map_err(wr)
}

fn attr_value(a: &RawAttr) -> MpasResult<NcAttrValue> {
    Ok(match a.nc_type {
        raw::NC_CHAR => NcAttrValue::Text(String::from_utf8(a.value.clone()).map_err(|_| {
            MpasError::Refusal(format!(
                "attribute {} carries non-UTF-8 text; the classic writer takes a \
                 String and would alter the bytes",
                a.name
            ))
        })?),
        raw::NC_INT => NcAttrValue::Ints(
            a.value
                .chunks_exact(4)
                .map(|c| i32::from_be_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        raw::NC_FLOAT => NcAttrValue::Floats(
            a.value
                .chunks_exact(4)
                .map(|c| f32::from_be_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        raw::NC_DOUBLE => NcAttrValue::Doubles(
            a.value
                .chunks_exact(8)
                .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        other => {
            return Err(MpasError::Refusal(format!(
                "attribute {} is nc_type {other}, which the classic writer has no \
                 value variant for; its bytes cannot be reproduced",
                a.name
            )))
        }
    })
}

/// The native tool's `create_graph_file`, from the REGIONAL arrays: first
/// line `nCells nEdgesInterior`, then per cell its present neighbours, each
/// followed by one space, LF line endings.
#[allow(clippy::too_many_arguments)]
fn write_graph_info(
    path: &Path,
    cell_ids: &[usize],
    edge_ids: &[usize],
    n_edges_on_cell: &[i32],
    cells_on_cell: &[i32],
    cells_on_edge: &[i32],
    cell_map: &[i32],
    max_edges: usize,
) -> MpasResult<()> {
    let mut interior = 0usize;
    for &e in edge_ids {
        let a = remap(cells_on_edge[e * 2], cell_map, "cellsOnEdge")?;
        let b = remap(cells_on_edge[e * 2 + 1], cell_map, "cellsOnEdge")?;
        if a > 0 && b > 0 {
            interior += 1;
        }
    }
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(format!("{} {}\n", cell_ids.len(), interior).as_bytes());
    for &c in cell_ids {
        for s in 0..n_edges_on_cell[c] as usize {
            let v = remap(cells_on_cell[c * max_edges + s], cell_map, "cellsOnCell")?;
            if v > 0 {
                out.extend_from_slice(v.to_string().as_bytes());
                out.push(b' ');
            }
        }
        out.push(b'\n');
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(&out)?;
    f.flush()?;
    Ok(())
}
