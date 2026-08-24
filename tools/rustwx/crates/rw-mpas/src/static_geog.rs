//! Bounded WPS_GEOG reader used by `rw_mpas_static`.
//!
//! Unlike a mosaic-based reader this never materializes an entire source
//! raster.  A tile plane is read, decoded, consumed and dropped before the
//! next plane is opened.  Peak source memory is therefore O(tile_x*tile_y),
//! independent of the size of the WPS_GEOG tree.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::{MpasError, MpasResult};
use crate::static_memory::checked_vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Continuous,
    Categorical,
}

#[derive(Debug, Clone)]
pub struct GeogIndex {
    pub kind: SourceKind,
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
    pub missing_value: Option<i64>,
    pub category_min: Option<i64>,
    pub category_max: Option<i64>,
    pub iswater: Option<i64>,
    pub islake: Option<i64>,
    pub isice: Option<i64>,
    pub isoilwater: Option<i64>,
    pub mminlu: String,
    pub row_order_top_bottom: bool,
}

impl GeogIndex {
    pub fn nz(&self) -> usize {
        (self.tile_z_end - self.tile_z_start + 1) as usize
    }

    pub fn full_tile_nx(&self) -> usize {
        (self.tile_x + 2 * self.tile_bdr) as usize
    }

    pub fn full_tile_ny(&self) -> usize {
        (self.tile_y + 2 * self.tile_bdr) as usize
    }

    pub fn plane_words(&self) -> usize {
        self.full_tile_nx() * self.full_tile_ny()
    }
}

#[derive(Debug, Clone)]
pub struct TileRef {
    pub xs: i64,
    pub xe: i64,
    pub ys: i64,
    pub ye: i64,
    pub path: PathBuf,
}

#[derive(Debug)]
pub struct GeogDataset {
    pub path: PathBuf,
    pub index: GeogIndex,
    pub tiles: Vec<TileRef>,
    pub nx_global: i64,
    pub ny_global: i64,
    pub wraps_x: bool,
}

/// Everything about a dataset that decides where a source pixel lands.
///
/// Two datasets with equal geometry map an identically-named tile to
/// identically-placed latitudes and longitudes, so work keyed on this is
/// reusable between them.  The dataset *name* is deliberately not part of it:
/// keying a cache on the name would recompute identical answers for every one
/// of the 30-arcsec products, which is the cost this exists to remove.
///
/// Floating-point members are keyed on their bit patterns, so two indices that
/// spell `known_lat` to different precisions -- as WPS_GEOG's own products do
/// -- are correctly treated as different geometries rather than merged.
/// `tile_bdr`, `wordsize`, `row_order` and the category tables are absent
/// because none of them move a pixel: the border is stripped on read and the
/// row flip restores the same bottom-to-top order the map already assumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceGeometry {
    dx: u64,
    dy: u64,
    known_x: u64,
    known_y: u64,
    known_lat: u64,
    known_lon: u64,
    tile_x: i64,
    tile_y: i64,
    nx_global: i64,
    ny_global: i64,
    wraps_x: bool,
}

fn raw_index(path: &Path) -> MpasResult<BTreeMap<String, String>> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        MpasError::Refusal(format!("WPS_GEOG index {} is unreadable: {e}", path.display()))
    })?;
    let mut out = BTreeMap::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let mut v = v.trim().to_string();
        if v.len() >= 2 {
            let bytes = v.as_bytes();
            if (bytes[0] == b'\'' && bytes[v.len() - 1] == b'\'')
                || (bytes[0] == b'"' && bytes[v.len() - 1] == b'"')
            {
                v = v[1..v.len() - 1].to_string();
            }
        }
        out.insert(k.trim().to_ascii_lowercase(), v);
    }
    Ok(out)
}

fn parse_f64(map: &BTreeMap<String, String>, key: &str) -> MpasResult<Option<f64>> {
    match map.get(key) {
        None => Ok(None),
        Some(v) => v.parse::<f64>().map(Some).map_err(|_| {
            MpasError::Refusal(format!("WPS_GEOG index has non-numeric {key}={v:?}"))
        }),
    }
}

fn parse_i64(map: &BTreeMap<String, String>, key: &str) -> MpasResult<Option<i64>> {
    Ok(parse_f64(map, key)?.map(|v| v.trunc() as i64))
}

fn need_f64(map: &BTreeMap<String, String>, key: &str) -> MpasResult<f64> {
    parse_f64(map, key)?.ok_or_else(|| {
        MpasError::Refusal(format!("WPS_GEOG index lacks required key '{key}'"))
    })
}

fn truthy(v: Option<&String>) -> bool {
    v.map(|s| matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | ".true."))
        .unwrap_or(false)
}

fn tile_name(name: &str) -> Option<(i64, i64, i64, i64)> {
    let (x, y) = name.split_once('.')?;
    if y.contains('.') {
        return None;
    }
    let (xs, xe) = x.split_once('-')?;
    let (ys, ye) = y.split_once('-')?;
    if xe.contains('-') || ye.contains('-') {
        return None;
    }
    let valid = |s: &str| !s.is_empty() && s.len() <= 6 && s.bytes().all(|b| b.is_ascii_digit());
    if ![xs, xe, ys, ye].iter().all(|s| valid(s)) {
        return None;
    }
    Some((xs.parse().ok()?, xe.parse().ok()?, ys.parse().ok()?, ye.parse().ok()?))
}

impl GeogDataset {
    pub fn open(path: &Path) -> MpasResult<Self> {
        let path = path.to_path_buf();
        let map = raw_index(&path.join("index"))?;
        let kind = match map.get("type").map(|s| s.to_ascii_lowercase()) {
            None => SourceKind::Continuous,
            Some(s) if s == "continuous" => SourceKind::Continuous,
            Some(s) if s == "categorical" => SourceKind::Categorical,
            Some(s) => {
                return Err(MpasError::Refusal(format!(
                    "WPS_GEOG dataset {} declares unsupported type {s:?}",
                    path.display()
                )))
            }
        };
        let projection = map
            .get("projection")
            .cloned()
            .unwrap_or_else(|| "regular_ll".to_string())
            .to_ascii_lowercase();
        if projection != "regular_ll" {
            return Err(MpasError::Refusal(format!(
                "WPS_GEOG dataset {} projection {projection:?} is unsupported by \
                 the bounded MPAS static path; preproject the source to regular_ll",
                path.display()
            )));
        }
        let (z0, z1) = if let Some(z) = parse_i64(&map, "tile_z")? {
            (1, z)
        } else {
            (
                parse_i64(&map, "tile_z_start")?.unwrap_or(1),
                parse_i64(&map, "tile_z_end")?.unwrap_or(1),
            )
        };
        if z1 < z0 {
            return Err(MpasError::Refusal(format!(
                "WPS_GEOG dataset {} has tile_z_end {z1} before tile_z_start {z0}",
                path.display()
            )));
        }
        let wordsize = need_f64(&map, "wordsize")?.trunc() as i64;
        if !matches!(wordsize, 1 | 2 | 4) {
            return Err(MpasError::Refusal(format!(
                "WPS_GEOG dataset {} wordsize {wordsize} is unsupported",
                path.display()
            )));
        }
        let idx = GeogIndex {
            kind,
            projection,
            dx: need_f64(&map, "dx")?,
            dy: need_f64(&map, "dy")?,
            known_x: parse_f64(&map, "known_x")?.unwrap_or(1.0),
            known_y: parse_f64(&map, "known_y")?.unwrap_or(1.0),
            known_lat: need_f64(&map, "known_lat")?,
            known_lon: need_f64(&map, "known_lon")?,
            wordsize: wordsize as u8,
            tile_x: need_f64(&map, "tile_x")?.trunc() as i64,
            tile_y: need_f64(&map, "tile_y")?.trunc() as i64,
            tile_z_start: z0,
            tile_z_end: z1,
            tile_bdr: parse_i64(&map, "tile_bdr")?.unwrap_or(0),
            signed: truthy(map.get("signed")),
            big_endian: map
                .get("endian")
                .map(|v| v.trim().eq_ignore_ascii_case("big"))
                .unwrap_or(true),
            scale_factor: parse_f64(&map, "scale_factor")?.unwrap_or(1.0),
            missing_value: parse_f64(&map, "missing_value")?.map(|v| v.trunc() as i64),
            category_min: parse_i64(&map, "category_min")?,
            category_max: parse_i64(&map, "category_max")?,
            iswater: parse_i64(&map, "iswater")?,
            islake: parse_i64(&map, "islake")?,
            isice: parse_i64(&map, "isice")?,
            isoilwater: parse_i64(&map, "isoilwater")?,
            mminlu: map.get("mminlu").cloned().unwrap_or_default(),
            row_order_top_bottom: map
                .get("row_order")
                .map(|v| v.trim().eq_ignore_ascii_case("top_bottom"))
                .unwrap_or(false),
        };
        if idx.tile_x <= 0 || idx.tile_y <= 0 || idx.tile_bdr < 0 || idx.dx == 0.0 || idx.dy == 0.0 {
            return Err(MpasError::Refusal(format!(
                "WPS_GEOG dataset {} carries invalid tile/grid geometry",
                path.display()
            )));
        }

        let mut tiles = Vec::new();
        let mut max_x = 0i64;
        let mut max_y = 0i64;
        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let Some((xs, xe, ys, ye)) = tile_name(&name) else {
                continue;
            };
            if xe - xs + 1 != idx.tile_x || ye - ys + 1 != idx.tile_y {
                return Err(MpasError::Refusal(format!(
                    "WPS_GEOG tile {} spans {}x{} but index declares {}x{} interior",
                    entry.path().display(),
                    xe - xs + 1,
                    ye - ys + 1,
                    idx.tile_x,
                    idx.tile_y
                )));
            }
            max_x = max_x.max(xe);
            max_y = max_y.max(ye);
            tiles.push(TileRef {
                xs,
                xe,
                ys,
                ye,
                path: entry.path(),
            });
        }
        if tiles.is_empty() {
            return Err(MpasError::Refusal(format!(
                "WPS_GEOG dataset {} contains no XSTART-XEND.YSTART-YEND tiles",
                path.display()
            )));
        }
        tiles.sort_by_key(|t| (t.ys, t.xs));

        let inferred_nx = (360.0 / idx.dx.abs()).round() as i64;
        let inferred_ny = (180.0 / idx.dy.abs()).round() as i64;
        let declared_nx = parse_i64(&map, "global_nx")?
            .or(parse_i64(&map, "nx_global")?);
        let declared_ny = parse_i64(&map, "global_ny")?
            .or(parse_i64(&map, "ny_global")?);
        let nx_global = declared_nx.unwrap_or_else(|| max_x.max(inferred_nx));
        let ny_global = declared_ny.unwrap_or_else(|| max_y.max(inferred_ny));
        if nx_global < max_x || ny_global < max_y {
            return Err(MpasError::Refusal(format!(
                "WPS_GEOG dataset {} declares global {}x{} smaller than tile inventory {}x{}",
                path.display(),
                nx_global,
                ny_global,
                max_x,
                max_y
            )));
        }
        let wraps_x = ((nx_global as f64) * idx.dx.abs() - 360.0).abs() <= 1.0e-6 * 360.0;

        Ok(Self {
            path,
            index: idx,
            tiles,
            nx_global,
            ny_global,
            wraps_x,
        })
    }

    /// The identity a reusable tile map is keyed on.  See [`SourceGeometry`].
    pub fn geometry(&self) -> SourceGeometry {
        SourceGeometry {
            dx: self.index.dx.to_bits(),
            dy: self.index.dy.to_bits(),
            known_x: self.index.known_x.to_bits(),
            known_y: self.index.known_y.to_bits(),
            known_lat: self.index.known_lat.to_bits(),
            known_lon: self.index.known_lon.to_bits(),
            tile_x: self.index.tile_x,
            tile_y: self.index.tile_y,
            nx_global: self.nx_global,
            ny_global: self.ny_global,
            wraps_x: self.wraps_x,
        }
    }

    pub fn source_xy_to_latlon(&self, x: f64, y: f64) -> (f64, f64) {
        let lat = self.index.known_lat + (y - self.index.known_y) * self.index.dy;
        let lon = self.index.known_lon + (x - self.index.known_x) * self.index.dx;
        (lat.to_radians(), wrap_lon_deg(lon).to_radians())
    }

    pub fn latlon_rad_to_source_xy(&self, lat: f64, lon: f64) -> (f64, f64) {
        let lat_deg = lat.to_degrees();
        let lon_deg = lon.to_degrees();
        let x = if self.wraps_x {
            let d = (lon_deg - self.index.known_lon).rem_euclid(360.0);
            let mut x = self.index.known_x + d / self.index.dx;
            if x >= self.nx_global as f64 + 0.5 {
                x -= self.nx_global as f64;
            }
            x
        } else {
            let d = (lon_deg - self.index.known_lon + 180.0).rem_euclid(360.0) - 180.0;
            self.index.known_x + d / self.index.dx
        };
        let y = self.index.known_y + (lat_deg - self.index.known_lat) / self.index.dy;
        (x, y)
    }

    pub fn plane_bytes(&self) -> u64 {
        self.index.plane_words() as u64 * self.index.wordsize as u64
    }

    pub fn validate_tile_file(&self, tile: &TileRef) -> MpasResult<()> {
        let expect = self
            .plane_bytes()
            .checked_mul(self.index.nz() as u64)
            .ok_or_else(|| MpasError::Refusal("WPS_GEOG tile byte size overflow".to_string()))?;
        let size = std::fs::metadata(&tile.path)?.len();
        if size < expect {
            return Err(MpasError::Refusal(format!(
                "WPS_GEOG tile {} is truncated: {} bytes, expected at least {}",
                tile.path.display(),
                size,
                expect
            )));
        }
        if size == expect {
            return Ok(());
        }
        // Upstream permits undeclared *complete* trailing planes only when
        // every word is zero.  Validate that without loading them.
        if (size - expect) % self.plane_bytes() != 0 {
            return Err(MpasError::Refusal(format!(
                "WPS_GEOG tile {} has {} trailing bytes, not complete planes of {} bytes",
                tile.path.display(),
                size - expect,
                self.plane_bytes()
            )));
        }
        let mut file = File::open(&tile.path)?;
        file.seek(SeekFrom::Start(expect))?;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            if buf[..n].iter().any(|&b| b != 0) {
                return Err(MpasError::Refusal(format!(
                    "WPS_GEOG tile {} has nonzero data in undeclared trailing z planes",
                    tile.path.display()
                )));
            }
        }
        Ok(())
    }

    /// Read exactly one z plane.  No decoded plane is retained by the dataset.
    pub fn read_plane(&self, tile: &TileRef, z: usize) -> MpasResult<Vec<i64>> {
        if z >= self.index.nz() {
            return Err(MpasError::Refusal(format!(
                "WPS_GEOG {} requested z={z}, but dataset has {} plane(s)",
                self.path.display(),
                self.index.nz()
            )));
        }
        self.validate_tile_file(tile)?;
        let plane_words = self.index.plane_words();
        let bytes_len = plane_words
            .checked_mul(self.index.wordsize as usize)
            .ok_or_else(|| MpasError::Refusal("WPS_GEOG plane byte size overflow".to_string()))?;
        let mut bytes = checked_vec::<u8>(bytes_len, "geog-read-plane", "encoded tile plane")?;
        let mut file = File::open(&tile.path)?;
        let offset = (z as u64)
            .checked_mul(bytes_len as u64)
            .ok_or_else(|| MpasError::Refusal("WPS_GEOG plane offset overflow".to_string()))?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut bytes)?;

        let mut out = checked_vec::<i64>(plane_words, "geog-decode-plane", "decoded tile plane")?;
        let ws = self.index.wordsize as usize;
        for (k, slot) in out.iter_mut().enumerate() {
            let s = &bytes[k * ws..(k + 1) * ws];
            *slot = match (self.index.wordsize, self.index.signed, self.index.big_endian) {
                (1, false, _) => s[0] as i64,
                (1, true, _) => s[0] as i8 as i64,
                (2, false, true) => u16::from_be_bytes([s[0], s[1]]) as i64,
                (2, false, false) => u16::from_le_bytes([s[0], s[1]]) as i64,
                (2, true, true) => i16::from_be_bytes([s[0], s[1]]) as i64,
                (2, true, false) => i16::from_le_bytes([s[0], s[1]]) as i64,
                (4, false, true) => u32::from_be_bytes(s.try_into().unwrap()) as i64,
                (4, false, false) => u32::from_le_bytes(s.try_into().unwrap()) as i64,
                (4, true, true) => i32::from_be_bytes(s.try_into().unwrap()) as i64,
                (4, true, false) => i32::from_le_bytes(s.try_into().unwrap()) as i64,
                _ => unreachable!("wordsize validated"),
            };
        }
        drop(bytes);

        if self.index.row_order_top_bottom {
            let nx = self.index.full_tile_nx();
            let ny = self.index.full_tile_ny();
            for j in 0..ny / 2 {
                let other = ny - 1 - j;
                for i in 0..nx {
                    out.swap(j * nx + i, other * nx + i);
                }
            }
        }
        Ok(out)
    }

    /// Bytes one full-width row of this dataset costs at element size `elem`.
    pub fn band_row_bytes(&self, elem: usize) -> u64 {
        self.nx_global as u64 * elem as u64
    }

    /// A full-width horizontal band of rows, in this dataset's own 1-based
    /// source index frame, mapped through `map` as it is decoded.
    ///
    /// WHY A BAND AND NOT THE ARRAY. The 30-arc-second products are 43,200 x
    /// 21,600 -- 933 million samples each -- and the sub-grid orography
    /// statistics need a rectangular window of them around every cell. Holding
    /// the array puts a multi-gigabyte floor under a tool that has to run on a
    /// laptop; holding a band of rows bounds it at the band, which is what the
    /// admission gate is shown and what it grants.
    ///
    /// Rows outside `[1, ny_global]` and any absent tile are left at `fill`.
    /// One tile plane is decoded, copied and dropped before the next is
    /// opened, so the transient is a tile and never the band's own source.
    pub fn read_band_with<T: Copy + Default>(
        &self,
        y_lo: i64,
        y_hi: i64,
        z: usize,
        fill: T,
        map: impl Fn(i64) -> T,
    ) -> MpasResult<Vec<T>> {
        if y_hi < y_lo {
            return Err(MpasError::Refusal(format!(
                "WPS_GEOG {} band rows {y_lo}..{y_hi} are empty",
                self.path.display()
            )));
        }
        let rows = (y_hi - y_lo + 1) as usize;
        let nx = self.nx_global as usize;
        let len = rows.checked_mul(nx).ok_or_else(|| {
            MpasError::Refusal("WPS_GEOG band element count overflow".to_string())
        })?;
        let mut out = crate::static_memory::checked_vec_filled::<T>(
            len,
            fill,
            "geog-band",
            "full-width source band",
        )?;
        let ty = self.index.tile_y;
        let tx = self.index.tile_x;
        let bdr = self.index.tile_bdr as usize;
        let full_nx = self.index.full_tile_nx();
        for tile in &self.tiles {
            if tile.ye < y_lo.max(1) || tile.ys > y_hi.min(self.ny_global) {
                continue;
            }
            let plane = self.read_plane(tile, z)?;
            let j0 = tile.ys.max(y_lo).max(1);
            let j1 = tile.ye.min(y_hi).min(self.ny_global);
            let i0 = tile.xs.max(1);
            let i1 = tile.xe.min(self.nx_global);
            if j1 < j0 || i1 < i0 {
                continue;
            }
            for j in j0..=j1 {
                let src_row = (j - tile.ys) as usize + bdr;
                let dst_row = (j - y_lo) as usize;
                for i in i0..=i1 {
                    let src = src_row * full_nx + (i - tile.xs) as usize + bdr;
                    out[dst_row * nx + (i - 1) as usize] = map(plane[src]);
                }
            }
            drop(plane);
            let _ = (ty, tx);
        }
        Ok(out)
    }

    #[inline]
    pub fn raw_to_value(&self, raw: i64) -> Option<f64> {
        if self.index.missing_value == Some(raw) {
            None
        } else {
            Some(raw as f64 * self.index.scale_factor)
        }
    }
}

fn wrap_lon_deg(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

/// Synthetic WPS_GEOG trees for tests.
///
/// Every fixture is written from bytes this module owns, so the bounded reader
/// is tested against its own inputs and never against a staged WPS_GEOG tree
/// that may or may not exist on the machine running the suite.
#[cfg(test)]
pub(crate) mod fixture {
    use std::path::PathBuf;

    /// A two-by-two single-plane continuous tile, big-endian, no border.
    pub(crate) const MINIMAL_INDEX: &str = "\
type=continuous
projection=regular_ll
dx=1.0
dy=1.0
known_x=1.0
known_y=1.0
known_lat=-89.5
known_lon=-179.5
wordsize=2
tile_x=2
tile_y=2
tile_z=1
tile_bdr=0
endian=big
signed=no
scale_factor=1.0
";

    pub(crate) const TILE_NAME: &str = "00001-00002.00001-00002";

    /// Big-endian u16 words 1,2,3,4 in row-major order.
    pub(crate) fn plane_1234_be() -> Vec<u8> {
        vec![0, 1, 0, 2, 0, 3, 0, 4]
    }

    /// A scratch directory holding `index` plus the named tile files.
    pub(crate) fn dataset(label: &str, index: &str, tiles: &[(&str, Vec<u8>)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rw-mpas-geog-{}-{}",
            std::process::id(),
            label
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("fixture directory");
        std::fs::write(dir.join("index"), index).expect("fixture index");
        for (name, bytes) in tiles {
            std::fs::write(dir.join(name), bytes).expect("fixture tile");
        }
        dir
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;

    fn open(label: &str, index: &str, tiles: &[(&str, Vec<u8>)]) -> MpasResult<GeogDataset> {
        GeogDataset::open(&dataset(label, index, tiles))
    }

    #[test]
    fn an_index_and_one_tile_round_trip() {
        let ds = open(
            "roundtrip",
            MINIMAL_INDEX,
            &[(TILE_NAME, plane_1234_be())],
        )
        .expect("dataset opens");
        assert_eq!(ds.index.kind, SourceKind::Continuous);
        assert_eq!(ds.index.tile_x, 2);
        assert_eq!(ds.index.nz(), 1);
        assert_eq!(ds.tiles.len(), 1);
        assert_eq!(ds.plane_bytes(), 8);
        // dx=1 over 360 degrees, so the reader infers a wrapping global grid.
        assert_eq!(ds.nx_global, 360);
        assert_eq!(ds.ny_global, 180);
        assert!(ds.wraps_x);
        let plane = ds.read_plane(&ds.tiles[0], 0).expect("plane reads");
        assert_eq!(plane, vec![1, 2, 3, 4]);
    }

    #[test]
    fn little_endian_words_decode_differently_from_big() {
        let index = MINIMAL_INDEX.replace("endian=big", "endian=little");
        let ds = open("little", &index, &[(TILE_NAME, plane_1234_be())]).expect("opens");
        assert!(!ds.index.big_endian);
        let plane = ds.read_plane(&ds.tiles[0], 0).expect("plane reads");
        assert_eq!(plane, vec![256, 512, 768, 1024]);
    }

    #[test]
    fn top_bottom_row_order_flips_the_plane() {
        let index = format!("{MINIMAL_INDEX}row_order=top_bottom\n");
        let ds = open("roworder", &index, &[(TILE_NAME, plane_1234_be())]).expect("opens");
        assert!(ds.index.row_order_top_bottom);
        let plane = ds.read_plane(&ds.tiles[0], 0).expect("plane reads");
        assert_eq!(plane, vec![3, 4, 1, 2]);
    }

    #[test]
    fn only_coordinate_named_files_are_taken_as_tiles() {
        let ds = open(
            "names",
            MINIMAL_INDEX,
            &[
                (TILE_NAME, plane_1234_be()),
                ("README", b"not a tile".to_vec()),
                ("index.bak", b"not a tile".to_vec()),
            ],
        )
        .expect("opens");
        assert_eq!(ds.tiles.len(), 1);
        assert!(ds.tiles[0].path.ends_with(TILE_NAME));
    }

    #[test]
    fn a_missing_required_key_is_refused_by_name() {
        let index = MINIMAL_INDEX.replace("dx=1.0\n", "");
        let err = open("nodx", &index, &[(TILE_NAME, plane_1234_be())]).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("required key 'dx'"), "{text}");
    }

    #[test]
    fn an_unsupported_projection_is_refused_with_the_remedy() {
        let index = MINIMAL_INDEX.replace("projection=regular_ll", "projection=lambert");
        let err = open("proj", &index, &[(TILE_NAME, plane_1234_be())]).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("lambert"), "{text}");
        assert!(text.contains("preproject the source to regular_ll"), "{text}");
    }

    #[test]
    fn an_unsupported_wordsize_is_refused() {
        let index = MINIMAL_INDEX.replace("wordsize=2", "wordsize=3");
        let err = open("wordsize", &index, &[(TILE_NAME, vec![0; 12])]).unwrap_err();
        assert!(err.to_string().contains("wordsize 3"), "{err}");
    }

    #[test]
    fn a_tile_whose_span_disagrees_with_the_index_is_refused() {
        let err = open(
            "span",
            MINIMAL_INDEX,
            &[("00001-00003.00001-00002", vec![0; 12])],
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("spans 3x2"), "{text}");
        assert!(text.contains("declares 2x2"), "{text}");
    }

    #[test]
    fn a_directory_with_no_tiles_is_refused() {
        let err = open("empty", MINIMAL_INDEX, &[]).unwrap_err();
        assert!(
            err.to_string().contains("contains no XSTART-XEND.YSTART-YEND tiles"),
            "{err}"
        );
    }

    #[test]
    fn a_truncated_tile_is_refused_before_it_is_decoded() {
        let ds = open("truncated", MINIMAL_INDEX, &[(TILE_NAME, vec![0, 1, 0, 2])])
            .expect("opens");
        let err = ds.read_plane(&ds.tiles[0], 0).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("is truncated"), "{text}");
        assert!(text.contains("expected at least 8"), "{text}");
    }

    #[test]
    fn nonzero_undeclared_trailing_planes_are_refused() {
        let mut bytes = plane_1234_be();
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 9]);
        let ds = open("trailing-nonzero", MINIMAL_INDEX, &[(TILE_NAME, bytes)]).expect("opens");
        let err = ds.read_plane(&ds.tiles[0], 0).unwrap_err();
        assert!(
            err.to_string().contains("nonzero data in undeclared trailing z planes"),
            "{err}"
        );
    }

    #[test]
    fn a_zero_filled_trailing_plane_is_accepted() {
        let mut bytes = plane_1234_be();
        bytes.extend_from_slice(&[0u8; 8]);
        let ds = open("trailing-zero", MINIMAL_INDEX, &[(TILE_NAME, bytes)]).expect("opens");
        let plane = ds.read_plane(&ds.tiles[0], 0).expect("plane reads");
        assert_eq!(plane, vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_partial_trailing_plane_is_refused_as_incomplete() {
        let mut bytes = plane_1234_be();
        bytes.extend_from_slice(&[0u8; 3]);
        let ds = open("trailing-partial", MINIMAL_INDEX, &[(TILE_NAME, bytes)]).expect("opens");
        let err = ds.read_plane(&ds.tiles[0], 0).unwrap_err();
        assert!(err.to_string().contains("not complete planes"), "{err}");
    }

    #[test]
    fn two_datasets_on_the_same_grid_share_one_geometry_key() {
        // The 30-arcsec landuse and soil-texture products differ in type,
        // wordsize, category tables and depth, and place every pixel in the
        // same place.  Work keyed on the geometry has to be shared between
        // them or it is recomputed once per product.
        let a = open("geom-a", MINIMAL_INDEX, &[(TILE_NAME, plane_1234_be())]).expect("opens");
        let categorical = format!(
            "{}\n",
            MINIMAL_INDEX
                .replace("type=continuous", "type=categorical")
                .trim_end()
        ) + "category_min=1\ncategory_max=4\n";
        let b = open("geom-b", &categorical, &[(TILE_NAME, plane_1234_be())]).expect("opens");
        assert_eq!(b.index.kind, SourceKind::Categorical);
        assert_eq!(a.geometry(), b.geometry());
    }

    #[test]
    fn a_shifted_origin_is_a_different_geometry_even_in_the_last_digit() {
        // topo_gmted2010_30s and modis_landuse_30s are both 1200x1200 at
        // 30 arcsec and their origins differ; greenfrac spells the same
        // latitude to one more decimal.  Merging either pair would put every
        // source pixel of one product into the wrong destination cell.
        let a = open("geom-origin-a", MINIMAL_INDEX, &[(TILE_NAME, plane_1234_be())])
            .expect("opens");
        for changed in [
            MINIMAL_INDEX.replace("known_lon=-179.5", "known_lon=-179.4999999"),
            MINIMAL_INDEX.replace("known_lat=-89.5", "known_lat=-89.50000001"),
            MINIMAL_INDEX.replace("dx=1.0", "dx=1.0000001"),
        ] {
            let b = open("geom-origin-b", &changed, &[(TILE_NAME, plane_1234_be())])
                .expect("opens");
            assert_ne!(a.geometry(), b.geometry(), "index was {changed}");
        }
    }

    #[test]
    fn the_border_and_the_row_order_do_not_change_the_geometry_key() {
        // Neither moves a pixel: the border is stripped on read, and the row
        // flip restores the bottom-to-top order the map already assumes.
        let a = open("geom-plain", MINIMAL_INDEX, &[(TILE_NAME, plane_1234_be())])
            .expect("opens");
        let flipped = format!("{MINIMAL_INDEX}row_order=top_bottom\n");
        let b = open("geom-flipped", &flipped, &[(TILE_NAME, plane_1234_be())]).expect("opens");
        assert_eq!(a.geometry(), b.geometry());
    }

    #[test]
    fn a_plane_past_the_declared_depth_is_refused() {
        let ds = open("depth", MINIMAL_INDEX, &[(TILE_NAME, plane_1234_be())]).expect("opens");
        let err = ds.read_plane(&ds.tiles[0], 1).unwrap_err();
        assert!(err.to_string().contains("requested z=1"), "{err}");
    }
}
