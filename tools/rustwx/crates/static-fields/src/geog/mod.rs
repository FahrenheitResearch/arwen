//! LANE 2.  WPS_GEOG dataset readers (`gpuwm/static/geog.py`).
//!
//! Each dataset directory is self-describing: an ASCII `index` file
//! (layout, value semantics, regular lat/lon georeferencing) plus flat
//! binary tiles named `XSTART-XEND.YSTART-YEND` (1-based inclusive, x
//! fastest, row 1 first, big-endian default -- write_geogrid.c).
//! Conventions carried over exactly (all arbitrated in the Python
//! docstrings): missing_value compared on RAW integers before
//! scale_factor; tile_bdr cropped for mosaics but retained for the
//! tile-local interpolation path; index authoritative for tile_z over
//! all-zero trailing padding (nonzero surplus refused); negative dy =
//! georeferencing only; 360-degree tile span => x wrap; staged
//! (footprint-minimized) trees keep global geometry via the
//! staged-inventory / declared global_nx rules; sparse declarations
//! never excuse tiles a model window needs (coverage gate).
//!
//! Documented divergences from the Python (never-bit-exact-to-a-bug):
//! an index missing a required key (`dx`, `wordsize`, ...) or carrying
//! an unrecognized `type` raises a named refusal here where the Python
//! would crash later with a bare `TypeError`/`KeyError`.  Array results
//! and refusal DECISIONS are unchanged.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::error::{Result, StaticError};

const TRUE_STRINGS: [&str; 4] = ["yes", "true", ".true.", "1"];

/// `np.mod` (result carries the divisor's sign; exact-zero results take
/// the divisor's sign as numpy's `npy_divmod` does).
pub(crate) fn pymod(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r == 0.0 {
        return f64::copysign(0.0, b);
    }
    if (r < 0.0) != (b < 0.0) { r + b } else { r }
}

/// Read an index file into a {lowercased key: unquoted value} map
/// (`_raw_index`).
pub(crate) fn raw_index(path: &Path) -> Result<BTreeMap<String, String>> {
    let text = std::fs::read_to_string(path).map_err(|err| {
        StaticError::Missing(format!(
            "WPS GEOG index {} is unreadable: {err}",
            path.display()
        ))
    })?;
    let mut out = BTreeMap::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() || !line.contains('=') {
            continue;
        }
        let (key, val) = line.split_once('=').expect("checked contains '='");
        let mut val = val.trim().to_string();
        let chars: Vec<char> = val.chars().collect();
        if chars.len() >= 2 {
            let first = chars[0];
            let last = *chars.last().expect("len >= 2");
            if (first == '\'' || first == '"') && last == first {
                val = chars[1..chars.len() - 1].iter().collect();
            }
        }
        out.insert(key.trim().to_lowercase(), val);
    }
    Ok(out)
}

fn parse_f64(path: &Path, key: &str, value: &str) -> Result<f64> {
    value.trim().parse::<f64>().map_err(|_| {
        StaticError::Invalid(format!(
            "WPS GEOG index {} has non-numeric {key}={value:?}",
            path.display()
        ))
    })
}

/// Typed view of a WPS_GEOG `index` file (defaults per WPS geogrid).
#[derive(Debug, Clone, PartialEq)]
pub struct GeogIndex {
    pub kind: SourceType,
    pub projection: String,
    pub dx: f64,
    pub dy: f64,
    pub known_x: f64,
    pub known_y: f64,
    pub known_lat: f64,
    pub known_lon: f64,
    pub wordsize: u8,
    pub tile_x: i64,
    pub tile_y: i64,
    pub tile_z_start: i64,
    pub tile_z_end: i64,
    pub tile_bdr: i64,
    pub signed: bool,
    pub big_endian: bool,
    pub scale_factor: f64,
    pub missing_value: Option<f64>,
    pub category_min: Option<i64>,
    pub category_max: Option<i64>,
    pub units: String,
    pub description: String,
    pub mminlu: String,
    pub iswater: Option<i64>,
    pub islake: Option<i64>,
    pub isice: Option<i64>,
    pub isurban: Option<i64>,
    pub row_order_top_bottom: bool,
    pub interp_option: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceType {
    Continuous,
    Categorical,
}

impl GeogIndex {
    pub fn nz(&self) -> i64 {
        self.tile_z_end - self.tile_z_start + 1
    }

    /// Parse one `index` file (`parse_index`).  LANE 2.
    pub fn parse(path: &std::path::Path) -> Result<Self> {
        let kv = raw_index(path)?;
        let f = |key: &str| -> Result<Option<f64>> {
            match kv.get(key) {
                None => Ok(None),
                Some(value) => parse_f64(path, key, value).map(Some),
            }
        };
        // Python `int(float(v))`: parse as float, truncate toward zero.
        let i = |key: &str| -> Result<Option<i64>> {
            Ok(f(key)?.map(|value| value.trunc() as i64))
        };
        let require = |key: &str, value: Option<f64>| -> Result<f64> {
            value.ok_or_else(|| {
                StaticError::Invalid(format!(
                    "WPS GEOG index {} lacks required key '{key}'",
                    path.display()
                ))
            })
        };

        let kind = match kv.get("type").map(|s| s.to_lowercase()) {
            None => SourceType::Continuous,
            Some(t) if t == "continuous" => SourceType::Continuous,
            Some(t) if t == "categorical" => SourceType::Categorical,
            Some(other) => {
                return Err(StaticError::Invalid(format!(
                    "WPS GEOG index {} declares unsupported type {other:?}",
                    path.display()
                )))
            }
        };
        let (z0, z1) = if kv.contains_key("tile_z") {
            (1, i("tile_z")?.expect("key present"))
        } else {
            (
                i("tile_z_start")?.unwrap_or(1),
                i("tile_z_end")?.unwrap_or(1),
            )
        };
        let wordsize = require("wordsize", f("wordsize")?)?.trunc() as i64;
        if !matches!(wordsize, 1 | 2 | 4) {
            return Err(StaticError::Invalid(format!(
                "WPS GEOG index {} declares unsupported wordsize {wordsize}",
                path.display()
            )));
        }
        Ok(GeogIndex {
            kind,
            projection: kv
                .get("projection")
                .map(|s| s.to_lowercase())
                .unwrap_or_else(|| "regular_ll".to_string()),
            dx: require("dx", f("dx")?)?,
            dy: require("dy", f("dy")?)?,
            known_x: f("known_x")?.unwrap_or(1.0),
            known_y: f("known_y")?.unwrap_or(1.0),
            known_lat: require("known_lat", f("known_lat")?)?,
            known_lon: require("known_lon", f("known_lon")?)?,
            wordsize: wordsize as u8,
            tile_x: require("tile_x", f("tile_x")?)?.trunc() as i64,
            tile_y: require("tile_y", f("tile_y")?)?.trunc() as i64,
            tile_z_start: z0,
            tile_z_end: z1,
            tile_bdr: i("tile_bdr")?.unwrap_or(0),
            signed: kv
                .get("signed")
                .map(|s| TRUE_STRINGS.contains(&s.trim().to_lowercase().as_str()))
                .unwrap_or(false),
            big_endian: kv
                .get("endian")
                .map(|s| s.trim().to_lowercase() == "big")
                .unwrap_or(true),
            scale_factor: f("scale_factor")?.unwrap_or(1.0),
            missing_value: f("missing_value")?,
            category_min: i("category_min")?,
            category_max: i("category_max")?,
            units: kv.get("units").cloned().unwrap_or_default(),
            description: kv.get("description").cloned().unwrap_or_default(),
            mminlu: kv.get("mminlu").cloned().unwrap_or_default(),
            iswater: i("iswater")?,
            islake: i("islake")?,
            isice: i("isice")?,
            isurban: i("isurban")?,
            row_order_top_bottom: kv
                .get("row_order")
                .map(|s| s.trim().to_lowercase() == "top_bottom")
                .unwrap_or(false),
            interp_option: kv
                .get("interp_option")
                .map(|s| s.trim().to_string())
                .unwrap_or_default(),
        })
    }
}

/// One decoded tile in processing order: `(nz, ny, nx)` raw integers
/// widened to i64.
#[derive(Debug)]
pub struct Tile {
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
    pub data: Vec<i64>,
}

/// A mosaicked window of source data in native storage: `raw[z][j][i]`
/// is source cell `(x0 + i, y0 + j)` (1-based, x wrap resolved).
/// `values(z)` applies missing masking (raw integers) then scale,
/// yielding f64 with NaN missing -- exactly `GeogWindow.values`.
#[derive(Debug, Clone)]
pub struct GeogWindow {
    pub index: GeogIndex,
    pub x0: i64,
    pub y0: i64,
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
    /// Raw integers widened to i64 (the widest classic word), storage
    /// order `(z, j, i)`.
    pub raw: Vec<i64>,
    /// Row-major `(ny, nx)` tile-presence coverage, when derived.
    pub coverage: Option<Vec<bool>>,
}

impl GeogWindow {
    /// Last covered x index, inclusive (`GeogWindow.x1`).
    pub fn x1(&self) -> i64 {
        self.x0 + self.nx as i64 - 1
    }

    /// Last covered y index, inclusive (`GeogWindow.y1`).
    pub fn y1(&self) -> i64 {
        self.y0 + self.ny as i64 - 1
    }

    /// Missing-masked, scaled plane as f64/NaN (`GeogWindow.values`).
    pub fn values(&self, z: usize) -> Vec<f64> {
        let n = self.ny * self.nx;
        let plane = &self.raw[z * n..(z + 1) * n];
        let missing = self.index.missing_value;
        let scale = self.index.scale_factor;
        plane
            .iter()
            .map(|&raw| {
                let mut v = raw as f64;
                if let Some(m) = missing {
                    if raw as f64 == m {
                        v = f64::NAN;
                    }
                }
                if scale != 1.0 {
                    v *= scale;
                }
                v
            })
            .collect()
    }
}

/// Cast a float fill value into the tile's storage integer exactly as
/// `np.full(..., fill, dtype=idx.dtype)` does: truncate toward zero,
/// then wrap into the word's width.
fn cast_fill(value: f64, wordsize: u8, signed: bool) -> i64 {
    let t = value.trunc() as i64;
    match (wordsize, signed) {
        (1, false) => (t as u8) as i64,
        (1, true) => (t as i8) as i64,
        (2, false) => (t as u16) as i64,
        (2, true) => (t as i16) as i64,
        (4, false) => (t as u32) as i64,
        (4, true) => (t as i32) as i64,
        _ => t,
    }
}

/// One WPS_GEOG dataset directory: index + tile inventory + windowing
/// (`GeogDataset`).
#[derive(Debug)]
pub struct GeogDataset {
    pub path: PathBuf,
    pub index: GeogIndex,
    pub declared_sparse: bool,
    pub tiles: BTreeMap<(i64, i64), PathBuf>,
    pub tile_inventory_bounds: (i64, i64, i64, i64),
    pub nx_global: i64,
    pub ny_global: i64,
    pub wraps_x: bool,
    pub extent_basis: &'static str,
    /// Decoded-tile cache, keyed `(xs, ys, include_border)` exactly like
    /// the Python `_tile_cache` (None = sparse absent tile).
    cache: Mutex<HashMap<(i64, i64, bool), Option<Arc<Tile>>>>,
}

/// `XSTART-XEND.YSTART-YEND` filename match (`_TILE_RE`): each field is
/// 1-6 ASCII digits, exactly one `-` per axis, exactly one `.`.
fn tile_name(name: &str) -> Option<(i64, i64, i64, i64)> {
    let (xpart, ypart) = name.split_once('.')?;
    if ypart.contains('.') {
        return None;
    }
    let field = |part: &str| -> Option<i64> {
        if part.is_empty()
            || part.len() > 6
            || !part.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        part.parse().ok()
    };
    let (xs, xe) = xpart.split_once('-')?;
    let (ys, ye) = ypart.split_once('-')?;
    if xe.contains('-') || ye.contains('-') {
        return None;
    }
    Some((field(xs)?, field(xe)?, field(ys)?, field(ye)?))
}

impl GeogDataset {
    /// Open + inventory a dataset directory (`GeogDataset.__init__`,
    /// including the staged-inventory extent rules).  LANE 2.
    pub fn open(path: &std::path::Path, sparse: Option<bool>) -> Result<Self> {
        let path = path.to_path_buf();
        let index_path = path.join("index");
        let index = GeogIndex::parse(&index_path)?;
        let raw = raw_index(&index_path)?;
        let declared = raw
            .get("sparse")
            .or_else(|| raw.get("is_sparse"))
            .map(|s| s.as_str())
            .unwrap_or("no");
        let declared_sparse = match sparse {
            Some(explicit) => explicit,
            None => TRUE_STRINGS.contains(&declared.trim().to_lowercase().as_str()),
        };
        if index.projection != "regular_ll" {
            return Err(StaticError::Invalid(format!(
                "projection '{}' not supported",
                index.projection
            )));
        }

        let mut tiles = BTreeMap::new();
        let mut xs_min: Option<i64> = None;
        let mut ys_min: Option<i64> = None;
        let mut xe_max = 0i64;
        let mut ye_max = 0i64;
        let entries = std::fs::read_dir(&path).map_err(|err| {
            StaticError::Missing(format!(
                "WPS GEOG dataset {} is unreadable: {err}",
                path.display()
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(StaticError::Io)?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some((xs, xe, ys, ye)) = tile_name(name) else {
                continue;
            };
            tiles.insert((xs, ys), entry.path());
            xs_min = Some(xs_min.map_or(xs, |v| v.min(xs)));
            ys_min = Some(ys_min.map_or(ys, |v| v.min(ys)));
            xe_max = xe_max.max(xe);
            ye_max = ye_max.max(ye);
        }
        if tiles.is_empty() {
            return Err(StaticError::Missing(format!(
                "no data tiles found in {}",
                path.display()
            )));
        }
        let xs_min = xs_min.expect("tiles nonempty");
        let ys_min = ys_min.expect("tiles nonempty");

        let declared_cells = |keys: &[&str]| -> Result<Option<i64>> {
            for key in keys {
                if let Some(value) = raw.get(*key) {
                    let value = parse_f64(&index_path, key, value)?.trunc() as i64;
                    if value <= 0 {
                        return Err(StaticError::Invalid(format!(
                            "WPS GEOG index {} declares non-positive {key}={value}",
                            index_path.display()
                        )));
                    }
                    return Ok(Some(value));
                }
            }
            Ok(None)
        };
        // Python `int(round(span / abs(spacing)))`: banker's rounding.
        let regular_ll_cells = |span: f64, spacing: f64| -> Option<i64> {
            if !spacing.is_finite() || spacing == 0.0 {
                return None;
            }
            let cells = (span / spacing.abs()).round_ties_even() as i64;
            if cells <= 0 {
                return None;
            }
            let tolerance = f64::max(1.0e-9, span * 2.0e-6);
            if (spacing.abs() * cells as f64 - span).abs() > tolerance {
                return None;
            }
            Some(cells)
        };

        let declared_nx = declared_cells(&["global_nx", "nx_global"])?;
        let declared_ny = declared_cells(&["global_ny", "ny_global"])?;
        let inferred_nx = regular_ll_cells(360.0, index.dx);
        let inferred_ny = regular_ll_cells(180.0, index.dy);
        let staged_inventory = declared_sparse || xs_min > 1 || ys_min > 1;

        if let Some(nx) = declared_nx {
            if nx < xe_max {
                return Err(StaticError::Invalid(format!(
                    "WPS GEOG global_nx={nx} is smaller than tile extent \
                     {xe_max} in {}",
                    path.display()
                )));
            }
        }
        if let Some(ny) = declared_ny {
            if ny < ye_max {
                return Err(StaticError::Invalid(format!(
                    "WPS GEOG global_ny={ny} is smaller than tile extent \
                     {ye_max} in {}",
                    path.display()
                )));
            }
        }

        let use_inferred_nx = inferred_nx.is_some_and(|nx| {
            nx >= xe_max && (staged_inventory || nx == xe_max)
        });
        let use_inferred_ny = inferred_ny.is_some_and(|ny| {
            ny >= ye_max && (staged_inventory || ny == ye_max)
        });
        let nx_global = declared_nx.unwrap_or(if use_inferred_nx {
            inferred_nx.expect("use_inferred_nx implies present")
        } else {
            xe_max
        });
        let ny_global = declared_ny.unwrap_or(if use_inferred_ny {
            inferred_ny.expect("use_inferred_ny implies present")
        } else {
            ye_max
        });
        let inferred_extent_expands_inventory = (use_inferred_nx
            && (xs_min > 1
                || inferred_nx.expect("use_inferred_nx implies present") > xe_max))
            || (use_inferred_ny
                && (ys_min > 1
                    || inferred_ny.expect("use_inferred_ny implies present")
                        > ye_max));
        let extent_basis = if declared_nx.is_some() || declared_ny.is_some() {
            "declared_global"
        } else if staged_inventory && inferred_extent_expands_inventory {
            "regular_ll_staged_inventory"
        } else if use_inferred_nx || use_inferred_ny {
            "regular_ll_complete"
        } else {
            "tile_inventory"
        };
        let wraps_x = inferred_nx.is_some_and(|nx| nx_global == nx);

        Ok(GeogDataset {
            path,
            index,
            declared_sparse,
            tiles,
            tile_inventory_bounds: (xs_min, xe_max, ys_min, ye_max),
            nx_global,
            ny_global,
            wraps_x,
            extent_basis,
            cache: Mutex::new(HashMap::new()),
        })
    }

    // -- georeferencing -----------------------------------------------------

    /// (lat, lon) -> fractional 1-based source coords (f64 path).
    pub fn latlon_to_xy(&self, lat: f64, lon: f64) -> (f64, f64) {
        let idx = &self.index;
        let x = if self.wraps_x {
            let d = pymod(lon - idx.known_lon, 360.0);
            let x = idx.known_x + d / idx.dx;
            if x >= self.nx_global as f64 + 0.5 {
                x - self.nx_global as f64
            } else {
                x
            }
        } else {
            let d = pymod(lon - idx.known_lon + 180.0, 360.0) - 180.0;
            idx.known_x + d / idx.dx
        };
        let y = idx.known_y + (lat - idx.known_lat) / idx.dy;
        (x, y)
    }

    /// Fractional source coords -> (lat, lon).
    pub fn xy_to_latlon(&self, x: f64, y: f64) -> (f64, f64) {
        let idx = &self.index;
        let lat = idx.known_lat + (y - idx.known_y) * idx.dy;
        let lon = idx.known_lon + (x - idx.known_x) * idx.dx;
        (lat, pymod(lon + 180.0, 360.0) - 180.0)
    }

    // -- tile IO ------------------------------------------------------------

    /// Decode one tile file's words to i64 by wordsize/sign/endianness.
    /// A trailing partial word is dropped, as `np.fromfile` drops it.
    fn decode_words(&self, bytes: &[u8]) -> Vec<i64> {
        let idx = &self.index;
        let ws = idx.wordsize as usize;
        let n = bytes.len() / ws;
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let chunk = &bytes[k * ws..(k + 1) * ws];
            let value = match (idx.wordsize, idx.signed, idx.big_endian) {
                (1, false, _) => chunk[0] as i64,
                (1, true, _) => chunk[0] as i8 as i64,
                (2, false, true) => {
                    u16::from_be_bytes([chunk[0], chunk[1]]) as i64
                }
                (2, false, false) => {
                    u16::from_le_bytes([chunk[0], chunk[1]]) as i64
                }
                (2, true, true) => {
                    i16::from_be_bytes([chunk[0], chunk[1]]) as i64
                }
                (2, true, false) => {
                    i16::from_le_bytes([chunk[0], chunk[1]]) as i64
                }
                (4, false, true) => u32::from_be_bytes(
                    chunk.try_into().expect("4-byte chunk"),
                ) as i64,
                (4, false, false) => u32::from_le_bytes(
                    chunk.try_into().expect("4-byte chunk"),
                ) as i64,
                (4, true, true) => i32::from_be_bytes(
                    chunk.try_into().expect("4-byte chunk"),
                ) as i64,
                (4, true, false) => i32::from_le_bytes(
                    chunk.try_into().expect("4-byte chunk"),
                ) as i64,
                _ => unreachable!("wordsize validated at parse"),
            };
            out.push(value);
        }
        out
    }

    /// Read tile with 1-based origin `(xs, ys)` (`_read_tile`): interior
    /// only by default, duplicated border halo retained with
    /// `include_border`.  `None` for a sparse missing tile.
    pub(crate) fn read_tile(
        &self,
        xs: i64,
        ys: i64,
        include_border: bool,
    ) -> Result<Option<Arc<Tile>>> {
        let key = (xs, ys, include_border);
        if let Some(hit) = self
            .cache
            .lock()
            .expect("tile cache poisoned")
            .get(&key)
        {
            return Ok(hit.clone());
        }
        let Some(file) = self.tiles.get(&(xs, ys)) else {
            self.cache
                .lock()
                .expect("tile cache poisoned")
                .insert(key, None);
            return Ok(None);
        };
        let idx = &self.index;
        let b = idx.tile_bdr;
        let ny = (idx.tile_y + 2 * b) as usize;
        let nx = (idx.tile_x + 2 * b) as usize;
        let nz = idx.nz() as usize;
        let bytes = std::fs::read(file)?;
        let mut raw = self.decode_words(&bytes);
        let plane_words = ny * nx;
        let expect = nz * plane_words;
        if raw.len() != expect {
            if raw.len() < expect || raw.len() % plane_words != 0 {
                return Err(StaticError::Invalid(format!(
                    "tile {} has {} words, expected {expect} \
                     ({nz}x{ny}x{nx})",
                    file.display(),
                    raw.len()
                )));
            }
            let actual_nz = raw.len() / plane_words;
            // Index remains authoritative over complete all-zero
            // trailing planes; nonzero surplus is refused with the first
            // offending element named (C order over the padding).
            for z in 0..(actual_nz - nz) {
                for j in 0..ny {
                    for i in 0..nx {
                        let value =
                            raw[(nz + z) * plane_words + j * nx + i];
                        if value != 0 {
                            return Err(StaticError::Invalid(format!(
                                "tile {} has {actual_nz} complete z planes \
                                 but its index declares {nz}; undeclared \
                                 trailing planes contain nonzero data \
                                 (first at z={}, y={}, x={})",
                                file.display(),
                                nz + z + 1,
                                j + 1,
                                i + 1
                            )));
                        }
                    }
                }
            }
            raw.truncate(expect);
        }
        let (mut data, mut out_ny, mut out_nx) = (raw, ny, nx);
        if b > 0 && !include_border {
            let bu = b as usize;
            let iny = ny - 2 * bu;
            let inx = nx - 2 * bu;
            let mut cropped = Vec::with_capacity(nz * iny * inx);
            for z in 0..nz {
                for j in bu..ny - bu {
                    let row = &data[z * plane_words + j * nx..];
                    cropped.extend_from_slice(&row[bu..bu + inx]);
                }
            }
            data = cropped;
            out_ny = iny;
            out_nx = inx;
        }
        if idx.row_order_top_bottom {
            let plane = out_ny * out_nx;
            let mut flipped = Vec::with_capacity(data.len());
            for z in 0..nz {
                for j in (0..out_ny).rev() {
                    let row = &data[z * plane + j * out_nx..];
                    flipped.extend_from_slice(&row[..out_nx]);
                }
            }
            data = flipped;
        }
        let tile = Arc::new(Tile {
            nz,
            ny: out_ny,
            nx: out_nx,
            data,
        });
        self.cache
            .lock()
            .expect("tile cache poisoned")
            .insert(key, Some(tile.clone()));
        Ok(Some(tile))
    }

    /// One native tile with its interpolation border retained
    /// (`read_tile_window`); `None` for a sparse absent tile.
    pub fn read_tile_window(
        &self,
        xs: i64,
        ys: i64,
    ) -> Result<Option<GeogWindow>> {
        let Some(tile) = self.read_tile(xs, ys, true)? else {
            return Ok(None);
        };
        let b = self.index.tile_bdr;
        Ok(Some(GeogWindow {
            index: self.index.clone(),
            x0: xs - b,
            y0: ys - b,
            nz: tile.nz,
            ny: tile.ny,
            nx: tile.nx,
            raw: tile.data.clone(),
            coverage: None,
        }))
    }

    fn tile_origin(&self, source: i64, tile_span: i64) -> i64 {
        (source - 1).div_euclid(tile_span) * tile_span + 1
    }

    /// Metadata-only boolean coverage for a window
    /// (`tile_coverage_mask`), row-major `(y1-y0+1, x1-x0+1)`.  LANE 2.
    pub fn tile_coverage_mask(
        &self,
        x0: i64,
        x1: i64,
        y0: i64,
        y1: i64,
    ) -> Result<Vec<bool>> {
        if x1 < x0 || y1 < y0 {
            return Err(StaticError::Invalid("empty window".to_string()));
        }
        let nxw = (x1 - x0 + 1) as usize;
        let nyw = (y1 - y0 + 1) as usize;
        let mut x_origin = Vec::with_capacity(nxw);
        let mut x_inside = Vec::with_capacity(nxw);
        for x in x0..=x1 {
            let source_x = if self.wraps_x {
                (x - 1).rem_euclid(self.nx_global) + 1
            } else {
                x
            };
            x_inside.push(self.wraps_x || (x >= 1 && x <= self.nx_global));
            x_origin.push(self.tile_origin(source_x, self.index.tile_x));
        }
        let mut y_origin = Vec::with_capacity(nyw);
        let mut y_inside = Vec::with_capacity(nyw);
        for y in y0..=y1 {
            y_inside.push(y >= 1 && y <= self.ny_global);
            y_origin.push(self.tile_origin(y, self.index.tile_y));
        }
        let mut present: BTreeMap<(i64, i64), bool> = BTreeMap::new();
        let mut mask = vec![false; nyw * nxw];
        for (j, (&yo, &yin)) in y_origin.iter().zip(&y_inside).enumerate() {
            for (i, (&xo, &xin)) in
                x_origin.iter().zip(&x_inside).enumerate()
            {
                let hit = *present
                    .entry((xo, yo))
                    .or_insert_with(|| self.tiles.contains_key(&(xo, yo)));
                mask[j * nxw + i] = hit && xin && yin;
            }
        }
        Ok(mask)
    }

    /// Cells of a requested window inside the dataset extent
    /// (`_extent_mask`), row-major bools.
    pub(crate) fn extent_mask(
        &self,
        x0: i64,
        x1: i64,
        y0: i64,
        y1: i64,
    ) -> Vec<bool> {
        let nxw = (x1 - x0 + 1) as usize;
        let nyw = (y1 - y0 + 1) as usize;
        let mut mask = vec![false; nyw * nxw];
        for (j, y) in (y0..=y1).enumerate() {
            let y_in = y >= 1 && y <= self.ny_global;
            for (i, x) in (x0..=x1).enumerate() {
                let x_in =
                    self.wraps_x || (x >= 1 && x <= self.nx_global);
                mask[j * nxw + i] = y_in && x_in;
            }
        }
        mask
    }

    /// Expected-but-absent tile origins (`missing_tiles`), ordered by
    /// `(y_origin, x_origin)` and returned as `(x, y)` pairs exactly as
    /// the Python does.  LANE 2.
    pub fn missing_tiles(
        &self,
        x0: i64,
        x1: i64,
        y0: i64,
        y1: i64,
    ) -> Result<Vec<(i64, i64)>> {
        let mask = self.tile_coverage_mask(x0, x1, y0, y1)?;
        let extent = self.extent_mask(x0, x1, y0, y1);
        let nxw = (x1 - x0 + 1) as usize;
        let mut origins: BTreeSet<(i64, i64)> = BTreeSet::new();
        for (k, (&covered, &inside)) in mask.iter().zip(&extent).enumerate() {
            if covered || !inside {
                continue;
            }
            let mut x = x0 + (k % nxw) as i64;
            let y = y0 + (k / nxw) as i64;
            if self.wraps_x {
                x = (x - 1).rem_euclid(self.nx_global) + 1;
            }
            origins.insert((
                self.tile_origin(y, self.index.tile_y),
                self.tile_origin(x, self.index.tile_x),
            ));
        }
        Ok(origins.into_iter().map(|(ys, xs)| (xs, ys)).collect())
    }

    /// All tile origins a window intersects (`required_tile_origins`),
    /// present and absent alike; metadata-only.
    pub fn required_tile_origins(
        &self,
        x0: i64,
        x1: i64,
        y0: i64,
        y1: i64,
    ) -> Result<Vec<(i64, i64)>> {
        if x1 < x0 || y1 < y0 {
            return Err(StaticError::Invalid("empty window".to_string()));
        }
        let mut x_origins: BTreeSet<i64> = BTreeSet::new();
        for x in x0..=x1 {
            let source_x = if self.wraps_x {
                (x - 1).rem_euclid(self.nx_global) + 1
            } else if x >= 1 && x <= self.nx_global {
                x
            } else {
                continue;
            };
            x_origins.insert(self.tile_origin(source_x, self.index.tile_x));
        }
        let mut y_origins: BTreeSet<i64> = BTreeSet::new();
        for y in y0..=y1 {
            if y >= 1 && y <= self.ny_global {
                y_origins.insert(self.tile_origin(y, self.index.tile_y));
            }
        }
        if x_origins.is_empty() || y_origins.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(x_origins.len() * y_origins.len());
        for &ys in &y_origins {
            for &xs in &x_origins {
                out.push((xs, ys));
            }
        }
        Ok(out)
    }

    /// Mosaic the (1-based, inclusive) index window (`read_window`).
    ///
    /// x wraps modulo the global width for global datasets; rows outside
    /// `[1, ny_global]` and absent tiles are filled with the dataset's
    /// missing value (or 0 when the index declares none).  Unexplained
    /// fill on a non-sparse tree is refused naming the absent tile.
    pub fn read_window(
        &self,
        x0: i64,
        x1: i64,
        y0: i64,
        y1: i64,
    ) -> Result<GeogWindow> {
        let idx = self.index.clone();
        if x1 < x0 || y1 < y0 {
            return Err(StaticError::Invalid("empty window".to_string()));
        }
        let nxw = (x1 - x0 + 1) as usize;
        let nyw = (y1 - y0 + 1) as usize;
        if nxw as i64 > self.nx_global {
            return Err(StaticError::Invalid(
                "window wider than the global grid".to_string(),
            ));
        }
        let coverage = self.tile_coverage_mask(x0, x1, y0, y1)?;
        let extent = self.extent_mask(x0, x1, y0, y1);
        if !self.declared_sparse {
            if let Some(k) = coverage
                .iter()
                .zip(&extent)
                .position(|(&covered, &inside)| !covered && inside)
            {
                let mut source_x = x0 + (k % nxw) as i64;
                let source_y = y0 + (k / nxw) as i64;
                if self.wraps_x {
                    source_x = (source_x - 1).rem_euclid(self.nx_global) + 1;
                }
                let tile_x = self.tile_origin(source_x, idx.tile_x);
                let tile_y = self.tile_origin(source_y, idx.tile_y);
                return Err(StaticError::Missing(format!(
                    "WPS GEOG dataset {} has unexplained fill at source \
                     index (x={source_x}, y={source_y}); expected tile \
                     origin ({tile_x}, {tile_y}) is absent. Declare this \
                     dataset sparse only when missing coverage is \
                     intentional.",
                    self.path.display()
                )));
            }
        }
        let fill = idx
            .missing_value
            .map(|m| cast_fill(m, idx.wordsize, idx.signed))
            .unwrap_or(0);
        let nz = idx.nz() as usize;
        let mut out = vec![fill; nz * nyw * nxw];

        // Split the x range into wrap-contiguous segments of absolute
        // coordinates.
        let mut segs: Vec<(i64, i64, i64)> = Vec::new();
        if self.wraps_x {
            let mut a = x0;
            while a <= x1 {
                let wa = (a - 1).rem_euclid(self.nx_global) + 1;
                let run = (x1 - a).min(self.nx_global - wa) + 1;
                segs.push((a, wa, run));
                a += run;
            }
        } else {
            segs.push((x0, x0, nxw as i64));
        }

        let (ty, tx) = (idx.tile_y, idx.tile_x);
        let yy0 = y0.max(1);
        let yy1 = y1.min(self.ny_global);
        let mut ys = (yy0 - 1).div_euclid(ty) * ty + 1;
        while ys <= yy1 {
            for &(abs_x0, wx0, run) in &segs {
                let wx1 = wx0 + run - 1;
                let mut xs = (wx0 - 1).div_euclid(tx) * tx + 1;
                while xs <= wx1 {
                    if let Some(tile) = self.read_tile(xs, ys, false)? {
                        // overlap in wrapped source coordinates
                        let ox0 = wx0.max(xs);
                        let ox1 = wx1.min(xs + tx - 1);
                        let oy0 = yy0.max(ys);
                        let oy1 = yy1.min(ys + ty - 1);
                        if ox1 >= ox0 && oy1 >= oy0 {
                            let di = ((abs_x0 - x0) + (ox0 - wx0)) as usize;
                            let dj = (oy0 - y0) as usize;
                            let cols = (ox1 - ox0 + 1) as usize;
                            let rows = (oy1 - oy0 + 1) as usize;
                            for z in 0..nz {
                                for r in 0..rows {
                                    let src = z * tile.ny * tile.nx
                                        + ((oy0 - ys) as usize + r) * tile.nx
                                        + (ox0 - xs) as usize;
                                    let dst = z * nyw * nxw
                                        + (dj + r) * nxw
                                        + di;
                                    out[dst..dst + cols].copy_from_slice(
                                        &tile.data[src..src + cols],
                                    );
                                }
                            }
                        }
                    }
                    xs += tx;
                }
            }
            ys += ty;
        }
        Ok(GeogWindow {
            index: idx,
            x0,
            y0,
            nz,
            ny: nyw,
            nx: nxw,
            raw: out,
            coverage: Some(coverage),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{
        assert_bits_f64, geog_root, golden_dir, hex_f64, json, read_bool,
        read_i64,
    };
    use serde_json::Value;

    fn check_index(idx: &GeogIndex, want: &Value, label: &str) {
        let kind = match idx.kind {
            SourceType::Continuous => "continuous",
            SourceType::Categorical => "categorical",
        };
        assert_eq!(kind, want["type"].as_str().unwrap(), "{label}: type");
        assert_eq!(
            idx.dx.to_bits(),
            hex_f64(&want["dx"]).to_bits(),
            "{label}: dx"
        );
        assert_eq!(
            idx.dy.to_bits(),
            hex_f64(&want["dy"]).to_bits(),
            "{label}: dy"
        );
        for (name, got, key) in [
            ("known_x", idx.known_x, "known_x"),
            ("known_y", idx.known_y, "known_y"),
            ("known_lat", idx.known_lat, "known_lat"),
            ("known_lon", idx.known_lon, "known_lon"),
            ("scale_factor", idx.scale_factor, "scale_factor"),
        ] {
            assert_eq!(
                got.to_bits(),
                hex_f64(&want[key]).to_bits(),
                "{label}: {name}"
            );
        }
        assert_eq!(
            idx.wordsize as i64,
            want["wordsize"].as_i64().unwrap(),
            "{label}: wordsize"
        );
        for (name, got, key) in [
            ("tile_x", idx.tile_x, "tile_x"),
            ("tile_y", idx.tile_y, "tile_y"),
            ("tile_z_start", idx.tile_z_start, "tile_z_start"),
            ("tile_z_end", idx.tile_z_end, "tile_z_end"),
            ("tile_bdr", idx.tile_bdr, "tile_bdr"),
        ] {
            assert_eq!(got, want[key].as_i64().unwrap(), "{label}: {name}");
        }
        assert_eq!(
            idx.signed,
            want["signed"].as_bool().unwrap(),
            "{label}: signed"
        );
        assert_eq!(
            idx.big_endian,
            want["endian_big"].as_bool().unwrap(),
            "{label}: endian"
        );
        assert_eq!(
            idx.row_order_top_bottom,
            want["row_order_top_bottom"].as_bool().unwrap(),
            "{label}: row_order"
        );
        match (&idx.missing_value, &want["missing_value"]) {
            (None, Value::Null) => {}
            (Some(m), v) => assert_eq!(
                m.to_bits(),
                hex_f64(v).to_bits(),
                "{label}: missing_value"
            ),
            (got, want) => {
                panic!("{label}: missing_value {got:?} vs {want:?}")
            }
        }
        for (name, got, key) in [
            ("category_min", idx.category_min, "category_min"),
            ("category_max", idx.category_max, "category_max"),
            ("iswater", idx.iswater, "iswater"),
            ("islake", idx.islake, "islake"),
            ("isice", idx.isice, "isice"),
            ("isurban", idx.isurban, "isurban"),
        ] {
            assert_eq!(got, want[key].as_i64(), "{label}: {name}");
        }
        assert_eq!(
            idx.mminlu,
            want["mminlu"].as_str().unwrap(),
            "{label}: mminlu"
        );
    }

    fn check_inventory(ds: &GeogDataset, want: &Value, label: &str) {
        assert_eq!(
            ds.declared_sparse,
            want["declared_sparse"].as_bool().unwrap(),
            "{label}: sparse"
        );
        assert_eq!(
            ds.tiles.len() as i64,
            want["tile_count"].as_i64().unwrap(),
            "{label}: tile count"
        );
        let bounds: Vec<i64> = want["tile_inventory_bounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(
            vec![
                ds.tile_inventory_bounds.0,
                ds.tile_inventory_bounds.1,
                ds.tile_inventory_bounds.2,
                ds.tile_inventory_bounds.3
            ],
            bounds,
            "{label}: bounds"
        );
        assert_eq!(
            ds.nx_global,
            want["nx_global"].as_i64().unwrap(),
            "{label}: nx_global"
        );
        assert_eq!(
            ds.ny_global,
            want["ny_global"].as_i64().unwrap(),
            "{label}: ny_global"
        );
        assert_eq!(
            ds.wraps_x,
            want["wraps_x"].as_bool().unwrap(),
            "{label}: wraps_x"
        );
        assert_eq!(
            ds.extent_basis,
            want["extent_basis"].as_str().unwrap(),
            "{label}: extent_basis"
        );
    }

    fn args4(v: &Value) -> (i64, i64, i64, i64) {
        let a: Vec<i64> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        (a[0], a[1], a[2], a[3])
    }

    fn check_window(
        ds: &GeogDataset,
        spec: &Value,
        exp_dir: &std::path::Path,
        label: &str,
    ) {
        let (x0, x1, y0, y1) = args4(&spec["args"]);
        let win = ds.read_window(x0, x1, y0, y1).unwrap_or_else(|err| {
            panic!("{label}: read_window refused: {err}")
        });
        assert_eq!(win.x0, spec["x0"].as_i64().unwrap_or(x0), "{label}: x0");
        assert_eq!(win.y0, spec["y0"].as_i64().unwrap_or(y0), "{label}: y0");
        let shape: Vec<usize> = spec["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(vec![win.nz, win.ny, win.nx], shape, "{label}: shape");
        let (_, raw) =
            read_i64(&exp_dir.join(spec["raw"].as_str().unwrap()));
        assert_eq!(win.raw, raw, "{label}: raw mosaic");
        if let Some(cov) = spec["coverage"].as_str() {
            let (_, want) = read_bool(&exp_dir.join(cov));
            assert_eq!(
                win.coverage.as_ref().expect("coverage"),
                &want,
                "{label}: coverage"
            );
        }
        if let Some(vals) = spec["values"].as_str() {
            let z = spec["values_z"].as_u64().unwrap_or(0) as usize;
            let (_, want) = crate::testsupport::read_f64(&exp_dir.join(vals));
            assert_bits_f64(&win.values(z), &want, &format!("{label}: values"));
        }
    }

    #[test]
    fn synthetic_datasets_match_the_python_reference() {
        let root = golden_dir();
        let exp_dir = root.join("synthetic_expected");
        let manifest = json(&exp_dir.join("manifest.json"));
        for (name, entry) in manifest.as_object().unwrap() {
            let ds_dir = root.join("synthetic").join(name);
            let ds = GeogDataset::open(&ds_dir, None)
                .unwrap_or_else(|err| panic!("{name}: open refused: {err}"));
            check_index(&ds.index, &entry["index"], name);
            check_inventory(&ds, &entry["inventory"], name);
            if entry["window"].is_object() {
                check_window(&ds, &entry["window"], &exp_dir, name);
            }
            if entry["tile_window"].is_object() {
                let spec = &entry["tile_window"];
                let a: Vec<i64> = spec["args"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect();
                let win = ds
                    .read_tile_window(a[0], a[1])
                    .unwrap()
                    .expect("tile present");
                assert_eq!(win.x0, spec["x0"].as_i64().unwrap());
                assert_eq!(win.y0, spec["y0"].as_i64().unwrap());
                let (_, raw) =
                    read_i64(&exp_dir.join(spec["raw"].as_str().unwrap()));
                assert_eq!(win.raw, raw, "{name}: tile window raw");
            }
            if let Some(spec) = entry.get("coverage_mask") {
                if spec.is_object() {
                    let (x0, x1, y0, y1) = args4(&spec["args"]);
                    let (_, want) = read_bool(
                        &exp_dir.join(spec["mask"].as_str().unwrap()),
                    );
                    assert_eq!(
                        ds.tile_coverage_mask(x0, x1, y0, y1).unwrap(),
                        want,
                        "{name}: coverage mask"
                    );
                }
            }
            if let Some(want) = entry.get("required_origins") {
                let (x0, x1, y0, y1) = if entry["window"].is_object() {
                    args4(&entry["window"]["args"])
                } else if let Some(spec) = entry.get("coverage_mask") {
                    args4(&spec["args"])
                } else {
                    unreachable!("required_origins without a window")
                };
                let got: Vec<Vec<i64>> = ds
                    .required_tile_origins(x0, x1, y0, y1)
                    .unwrap()
                    .into_iter()
                    .map(|(x, y)| vec![x, y])
                    .collect();
                let want: Vec<Vec<i64>> = want
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|pair| {
                        pair.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_i64().unwrap())
                            .collect()
                    })
                    .collect();
                assert_eq!(got, want, "{name}: required origins");
            }
            if let Some(want) = entry.get("missing_tiles") {
                let (x0, x1, y0, y1) = if entry["window"].is_object() {
                    args4(&entry["window"]["args"])
                } else if let Some(spec) = entry.get("coverage_mask") {
                    args4(&spec["args"])
                } else if entry["read_window_error"].is_object() {
                    args4(&entry["read_window_error"]["args"])
                } else {
                    unreachable!("missing_tiles without a window")
                };
                let got: Vec<Vec<i64>> = ds
                    .missing_tiles(x0, x1, y0, y1)
                    .unwrap()
                    .into_iter()
                    .map(|(x, y)| vec![x, y])
                    .collect();
                let want: Vec<Vec<i64>> = want
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|pair| {
                        pair.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_i64().unwrap())
                            .collect()
                    })
                    .collect();
                assert_eq!(got, want, "{name}: missing tiles");
            }
            for (key, pts) in [
                ("latlon_to_xy", entry.get("latlon_to_xy")),
                ("xy_to_latlon", entry.get("xy_to_latlon")),
            ] {
                let Some(pts) = pts.and_then(|p| p.as_array()) else {
                    continue;
                };
                for (k, pt) in pts.iter().enumerate() {
                    if key == "latlon_to_xy" {
                        let (x, y) = ds.latlon_to_xy(
                            hex_f64(&pt["lat"]),
                            hex_f64(&pt["lon"]),
                        );
                        assert_eq!(
                            x.to_bits(),
                            hex_f64(&pt["x"]).to_bits(),
                            "{name}: latlon_to_xy[{k}].x"
                        );
                        assert_eq!(
                            y.to_bits(),
                            hex_f64(&pt["y"]).to_bits(),
                            "{name}: latlon_to_xy[{k}].y"
                        );
                    } else {
                        let (lat, lon) = ds.xy_to_latlon(
                            hex_f64(&pt["x"]),
                            hex_f64(&pt["y"]),
                        );
                        assert_eq!(
                            lat.to_bits(),
                            hex_f64(&pt["lat"]).to_bits(),
                            "{name}: xy_to_latlon[{k}].lat"
                        );
                        assert_eq!(
                            lon.to_bits(),
                            hex_f64(&pt["lon"]).to_bits(),
                            "{name}: xy_to_latlon[{k}].lon"
                        );
                    }
                }
            }
            for spec in [entry.get("read_window_error"), entry.get("too_wide")]
                .into_iter()
                .flatten()
                .filter(|s| s.is_object())
            {
                let (x0, x1, y0, y1) = args4(&spec["args"]);
                let err = ds
                    .read_window(x0, x1, y0, y1)
                    .expect_err("refusal expected");
                let message = err.to_string();
                let needle = spec["contains"].as_str().unwrap();
                assert!(
                    message.contains(needle),
                    "{name}: refusal {message:?} lacks {needle:?}"
                );
            }
        }
    }

    #[test]
    fn real_indexes_and_windows_match_the_python_reference() {
        let Some(geog) = geog_root() else {
            eprintln!("SKIP: WPS_GEOG reference tree not present");
            return;
        };
        let root = golden_dir().join("real");
        let manifest = json(&root.join("manifest.json"));
        for (field, entry) in manifest["indexes"].as_object().unwrap() {
            let dir = geog.join(entry["dir"].as_str().unwrap());
            let ds = GeogDataset::open(&dir, None).unwrap_or_else(|err| {
                panic!("{field}: open refused: {err}")
            });
            check_index(&ds.index, &entry["index"], field);
            check_inventory(&ds, &entry["inventory"], field);
        }
        let topo = GeogDataset::open(
            &geog.join(
                manifest["indexes"]["terrain"]["dir"].as_str().unwrap(),
            ),
            None,
        )
        .unwrap();
        check_window(&topo, &manifest["topo_window"], &root, "topo");
        let soiltemp = GeogDataset::open(
            &geog.join(
                manifest["indexes"]["soil_temperature"]["dir"]
                    .as_str()
                    .unwrap(),
            ),
            None,
        )
        .unwrap();
        check_window(
            &soiltemp,
            &manifest["soiltemp_window"],
            &root,
            "soiltemp",
        );
    }
}
