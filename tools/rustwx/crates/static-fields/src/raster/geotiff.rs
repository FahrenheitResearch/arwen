//! LANE 3.  GeoTIFF decode/encode.
//!
//! Decode: classic and BigTIFF, striped and tiled, compression none /
//! deflate (8 and legacy 32946) / LZW (5), predictor none / horizontal
//! / floating-point, samples u8/i16/u16/i32/f32/f64, GeoTIFF
//! georeferencing (pixel scale + tiepoint or transform matrix), the
//! GDAL nodata tag, the CRS keys the closed [`super::Crs`] set covers,
//! plus an explicit override slot for sources whose files carry no
//! CRS.  Reads are windowed — only the strips/tiles a window touches
//! are fetched and inflated — so the multi-gigabyte CONUS land-cover
//! raster is read through the same seam as a one-degree tile.  Refuse
//! — by name — anything outside that envelope rather than misread it.
//!
//! Encode: the derived-window writer (`derive_terrain_window`,
//! `derive_global_terrain_window`, `derive_landcover_window`
//! replacements): single band, deflate, tiled 256x256, predictor 2
//! (integers) / 3 (floats), little-endian, fixed tag order —
//! byte-deterministic for identical inputs because the derivation
//! cache is keyed by content digest.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{Result, StaticError};
use crate::raster::{Crs, Raster};

fn invalid(message: impl Into<String>) -> StaticError {
    StaticError::Invalid(message.into())
}

// ---------------------------------------------------------------------------
// Sample formats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleType {
    U8,
    U16,
    I16,
    I32,
    F32,
    F64,
}

impl SampleType {
    fn bytes(self) -> usize {
        match self {
            SampleType::U8 => 1,
            SampleType::U16 | SampleType::I16 => 2,
            SampleType::I32 | SampleType::F32 => 4,
            SampleType::F64 => 8,
        }
    }

    fn from_tags(bits: u16, format: u16, path: &Path) -> Result<Self> {
        Ok(match (bits, format) {
            (8, 1) => SampleType::U8,
            (16, 1) => SampleType::U16,
            (16, 2) => SampleType::I16,
            (32, 2) => SampleType::I32,
            (32, 3) => SampleType::F32,
            (64, 3) => SampleType::F64,
            _ => {
                return Err(invalid(format!(
                    "GeoTIFF {path:?}: sample type ({bits} bits, \
                     SampleFormat {format}) is outside the substrate's \
                     decode envelope (u8/u16/i16/i32/f32/f64)"
                )))
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Low-level byte access
// ---------------------------------------------------------------------------

struct ByteReader {
    file: File,
    little_endian: bool,
}

impl ByteReader {
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; len];
        self.file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    fn u16_from(&self, raw: &[u8]) -> u16 {
        let pair = [raw[0], raw[1]];
        if self.little_endian {
            u16::from_le_bytes(pair)
        } else {
            u16::from_be_bytes(pair)
        }
    }

    fn u32_from(&self, raw: &[u8]) -> u32 {
        let quad = [raw[0], raw[1], raw[2], raw[3]];
        if self.little_endian {
            u32::from_le_bytes(quad)
        } else {
            u32::from_be_bytes(quad)
        }
    }

    fn u64_from(&self, raw: &[u8]) -> u64 {
        let oct: [u8; 8] = raw[..8].try_into().unwrap();
        if self.little_endian {
            u64::from_le_bytes(oct)
        } else {
            u64::from_be_bytes(oct)
        }
    }

    fn f64_from(&self, raw: &[u8]) -> f64 {
        f64::from_bits(self.u64_from(raw))
    }
}

/// One parsed IFD entry with its payload fully fetched.
#[derive(Debug, Clone)]
struct TagEntry {
    field_type: u16,
    count: u64,
    payload: Vec<u8>,
}

const TYPE_SIZES: [usize; 19] = [
    0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4, 0, 0, 8, 8, 8,
];

// ---------------------------------------------------------------------------
// The reader
// ---------------------------------------------------------------------------

/// One open single-band GeoTIFF, first IFD (full resolution; COG
/// overviews in later IFDs are deliberately ignored).
pub struct TiffReader {
    bytes: ByteReader,
    path: std::path::PathBuf,
    pub width: usize,
    pub height: usize,
    sample: SampleType,
    compression: u16,
    predictor: u16,
    /// (tile_width, tile_height) for tiled files; strips are treated
    /// as width x rows_per_strip tiles in a single column.
    block_w: usize,
    block_h: usize,
    tiled: bool,
    offsets: Vec<u64>,
    byte_counts: Vec<u64>,
    pub nodata: Option<f64>,
    pub transform: [f64; 6],
    pub crs: Option<Crs>,
}

impl TiffReader {
    pub fn open(path: &Path) -> Result<TiffReader> {
        let file = File::open(path).map_err(|err| {
            StaticError::Missing(format!(
                "high-resolution raster missing: {path:?} ({err})"
            ))
        })?;
        let mut bytes = ByteReader { file, little_endian: true };
        let header = bytes.read_at(0, 8)?;
        let little_endian = match &header[0..2] {
            b"II" => true,
            b"MM" => false,
            other => {
                return Err(invalid(format!(
                    "{path:?} is not a TIFF (byte-order mark {other:?})"
                )))
            }
        };
        bytes.little_endian = little_endian;
        let magic = bytes.u16_from(&header[2..4]);
        let (big, first_ifd) = match magic {
            42 => (false, bytes.u32_from(&header[4..8]) as u64),
            43 => {
                let more = bytes.read_at(4, 12)?;
                let offset_size = bytes.u16_from(&more[0..2]);
                if offset_size != 8 {
                    return Err(invalid(format!(
                        "{path:?}: BigTIFF offset size {offset_size} \
                         is not 8"
                    )));
                }
                (true, bytes.u64_from(&more[4..12]))
            }
            other => {
                return Err(invalid(format!(
                    "{path:?} is not a TIFF (magic {other})"
                )))
            }
        };

        let mut tags: std::collections::BTreeMap<u16, TagEntry> =
            std::collections::BTreeMap::new();
        let (entry_size, count_len) = if big { (20usize, 8usize) } else { (12usize, 2usize) };
        let count_raw = bytes.read_at(first_ifd, count_len)?;
        let entry_count = if big {
            bytes.u64_from(&count_raw)
        } else {
            bytes.u16_from(&count_raw) as u64
        };
        let table = bytes.read_at(
            first_ifd + count_len as u64,
            entry_size * entry_count as usize,
        )?;
        for index in 0..entry_count as usize {
            let raw = &table[index * entry_size..(index + 1) * entry_size];
            let tag = bytes.u16_from(&raw[0..2]);
            let field_type = bytes.u16_from(&raw[2..4]);
            let count = if big {
                bytes.u64_from(&raw[4..12])
            } else {
                bytes.u32_from(&raw[4..8]) as u64
            };
            let value_raw =
                if big { &raw[12..20] } else { &raw[8..12] };
            let type_size = *TYPE_SIZES
                .get(field_type as usize)
                .filter(|size| **size > 0)
                .unwrap_or(&1);
            let total = type_size * count as usize;
            let inline_cap = if big { 8 } else { 4 };
            let entry = if total <= inline_cap {
                TagEntry {
                    field_type,
                    count,
                    payload: value_raw[..total.min(inline_cap)].to_vec(),
                }
            } else {
                let offset = if big {
                    bytes.u64_from(value_raw)
                } else {
                    bytes.u32_from(value_raw) as u64
                };
                TagEntry {
                    field_type,
                    count,
                    payload: bytes.read_at(offset, total)?,
                }
            };
            tags.insert(tag, entry);
        }

        let get_ints = |tag: u16| -> Option<Vec<u64>> {
            let entry = tags.get(&tag)?;
            let size = TYPE_SIZES[entry.field_type as usize];
            let mut out = Vec::with_capacity(entry.count as usize);
            for index in 0..entry.count as usize {
                let raw = &entry.payload[index * size..(index + 1) * size];
                out.push(match entry.field_type {
                    1 | 2 | 6 | 7 => raw[0] as u64,
                    3 => bytes.u16_from(raw) as u64,
                    4 => bytes.u32_from(raw) as u64,
                    16 | 17 => bytes.u64_from(raw),
                    8 => bytes.u16_from(raw) as u64,
                    9 => bytes.u32_from(raw) as u64,
                    _ => return None,
                });
            }
            Some(out)
        };
        let get_doubles = |tag: u16| -> Option<Vec<f64>> {
            let entry = tags.get(&tag)?;
            if entry.field_type != 12 {
                return None;
            }
            Some(
                entry
                    .payload
                    .chunks_exact(8)
                    .map(|raw| bytes.f64_from(raw))
                    .collect(),
            )
        };
        let get_ascii = |tag: u16| -> Option<String> {
            let entry = tags.get(&tag)?;
            let text: Vec<u8> = entry
                .payload
                .iter()
                .copied()
                .take_while(|byte| *byte != 0)
                .collect();
            String::from_utf8(text).ok()
        };
        let first_int = |tag: u16| get_ints(tag).and_then(|v| v.first().copied());

        let width = first_int(256).ok_or_else(|| {
            invalid(format!("{path:?}: TIFF has no ImageWidth"))
        })? as usize;
        let height = first_int(257).ok_or_else(|| {
            invalid(format!("{path:?}: TIFF has no ImageLength"))
        })? as usize;
        let samples_per_pixel = first_int(277).unwrap_or(1);
        if samples_per_pixel != 1 {
            return Err(invalid(format!(
                "{path:?}: {samples_per_pixel} samples per pixel; the \
                 substrate reads single-band rasters only"
            )));
        }
        let bits = first_int(258).unwrap_or(1) as u16;
        let format = first_int(339).unwrap_or(1) as u16;
        let sample = SampleType::from_tags(bits, format, path)?;
        let compression = first_int(259).unwrap_or(1) as u16;
        if !matches!(compression, 1 | 5 | 8 | 32946) {
            return Err(invalid(format!(
                "{path:?}: TIFF compression {compression} is outside \
                 the substrate's decode envelope (none, LZW, deflate)"
            )));
        }
        let predictor = first_int(317).unwrap_or(1) as u16;
        if !matches!(predictor, 1 | 2 | 3) {
            return Err(invalid(format!(
                "{path:?}: TIFF predictor {predictor} is not \
                 none/horizontal/floating-point"
            )));
        }
        let planar = first_int(284).unwrap_or(1);
        if planar != 1 {
            return Err(invalid(format!(
                "{path:?}: planar configuration {planar} is not chunky"
            )));
        }

        let (tiled, block_w, block_h, offsets, byte_counts) =
            if tags.contains_key(&324) {
                let tile_w = first_int(322).ok_or_else(|| {
                    invalid(format!("{path:?}: tiled TIFF lacks TileWidth"))
                })? as usize;
                let tile_h = first_int(323).ok_or_else(|| {
                    invalid(format!("{path:?}: tiled TIFF lacks TileLength"))
                })? as usize;
                (
                    true,
                    tile_w,
                    tile_h,
                    get_ints(324).unwrap_or_default(),
                    get_ints(325).unwrap_or_default(),
                )
            } else {
                let rows = first_int(278).unwrap_or(height as u64) as usize;
                (
                    false,
                    width,
                    rows,
                    get_ints(273).unwrap_or_default(),
                    get_ints(279).unwrap_or_default(),
                )
            };
        if offsets.is_empty() || offsets.len() != byte_counts.len() {
            return Err(invalid(format!(
                "{path:?}: TIFF block offsets/counts are inconsistent \
                 ({} offsets, {} counts)",
                offsets.len(),
                byte_counts.len()
            )));
        }

        // Georeferencing: pixel scale + tiepoint, or the full matrix.
        let transform = if let Some(matrix) = get_doubles(34264) {
            if matrix.len() < 8 {
                return Err(invalid(format!(
                    "{path:?}: ModelTransformation carries \
                     {} values, expected 16",
                    matrix.len()
                )));
            }
            [matrix[0], matrix[1], matrix[3], matrix[4], matrix[5], matrix[7]]
        } else {
            let scale = get_doubles(33550).ok_or_else(|| {
                invalid(format!(
                    "{path:?}: GeoTIFF lacks both ModelPixelScale and \
                     ModelTransformation"
                ))
            })?;
            let tie = get_doubles(33922).ok_or_else(|| {
                invalid(format!("{path:?}: GeoTIFF lacks ModelTiepoint"))
            })?;
            if scale.len() < 2 || tie.len() < 6 {
                return Err(invalid(format!(
                    "{path:?}: GeoTIFF pixel scale/tiepoint are too short"
                )));
            }
            let (i, j, x, y) = (tie[0], tie[1], tie[3], tie[4]);
            [
                scale[0],
                0.0,
                x - i * scale[0],
                0.0,
                -scale[1],
                y + j * scale[1],
            ]
        };

        // GTRasterTypeGeoKey = RasterPixelIsPoint: the tiepoint names a
        // pixel CENTRE, not the raster's upper-left corner, so the
        // origin is half a pixel north-west of it.  This is the
        // GeoTIFF spec's rule and what GDAL applies; skipping it
        // georeferences the whole raster half a pixel south-east.
        //
        // It is not a theoretical case.  Copernicus DEM GLO-30 -- the
        // near-global terrain source `terrain_source = "auto"` picks
        // outside the United States -- ships PixelIsPoint with a
        // tiepoint at (8.0, 47.0) for the N46/E008 tile.  Read without
        // the shift, the mosaic samples the wrong 30 m source pixel
        // wherever the half-pixel crosses a pixel boundary, and the
        // warp onto the model grid carries the bias through: MEASURED
        // on a real 500 m Alpine domain, max |delta| 62.2 m and mean
        // |delta| 8.05 m of terrain height against the rasterio path.
        //
        // The committed lane-3 goldens could not catch this: they were
        // written out of the source tiles BY RASTERIO, which emits
        // PixelIsArea with the shift already folded into the tiepoint,
        // so the fixtures never carried the tag under test.
        let transform = if raster_pixel_is_point(&tags, &bytes) {
            [
                transform[0],
                transform[1],
                transform[2] - 0.5 * transform[0] - 0.5 * transform[1],
                transform[3],
                transform[4],
                transform[5] - 0.5 * transform[3] - 0.5 * transform[4],
            ]
        } else {
            transform
        };

        let nodata = get_ascii(42113).and_then(|text| {
            let trimmed = text.trim().to_ascii_lowercase();
            if trimmed == "nan" {
                Some(f64::NAN)
            } else {
                trimmed.parse::<f64>().ok()
            }
        });

        let crs = parse_geokeys(path, &tags, &bytes)?;

        Ok(TiffReader {
            bytes,
            path: path.to_path_buf(),
            width,
            height,
            sample,
            compression,
            predictor,
            block_w,
            block_h,
            tiled,
            offsets,
            byte_counts,
            nodata,
            transform,
            crs,
        })
    }

    /// Decode one block (tile or strip) into raw sample bytes.
    fn block_bytes(&mut self, index: usize, rows_in_block: usize) -> Result<Vec<u8>> {
        let offset = self.offsets[index];
        let count = self.byte_counts[index] as usize;
        let compressed = self.bytes.read_at(offset, count)?;
        let expected =
            self.block_w * rows_in_block * self.sample.bytes();
        let mut raw = match self.compression {
            1 => compressed,
            8 | 32946 => miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(
                &compressed,
                self.block_w * self.block_h * self.sample.bytes(),
            )
            .map_err(|err| {
                invalid(format!(
                    "{:?}: deflate block {index} failed to inflate: {err}",
                    self.path
                ))
            })?,
            5 => {
                let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(
                    weezl::BitOrder::Msb,
                    8,
                );
                decoder.decode(&compressed).map_err(|err| {
                    invalid(format!(
                        "{:?}: LZW block {index} failed to decode: {err}",
                        self.path
                    ))
                })?
            }
            other => {
                return Err(invalid(format!(
                    "{:?}: unsupported compression {other}",
                    self.path
                )))
            }
        };
        if raw.len() < expected {
            return Err(invalid(format!(
                "{:?}: block {index} decoded to {} bytes, expected \
                 at least {expected}",
                self.path,
                raw.len()
            )));
        }
        raw.truncate(expected);
        self.undo_predictor(&mut raw, rows_in_block);
        Ok(raw)
    }

    fn undo_predictor(&self, raw: &mut [u8], rows: usize) {
        let sample_bytes = self.sample.bytes();
        let row_samples = self.block_w;
        let row_bytes = row_samples * sample_bytes;
        match self.predictor {
            1 => {}
            2 => {
                for row in 0..rows {
                    let start = row * row_bytes;
                    match self.sample {
                        SampleType::U8 => {
                            for i in 1..row_samples {
                                raw[start + i] =
                                    raw[start + i].wrapping_add(raw[start + i - 1]);
                            }
                        }
                        SampleType::U16 | SampleType::I16 => {
                            let mut prev = u16::from_le_bytes([
                                raw[start],
                                raw[start + 1],
                            ]);
                            for i in 1..row_samples {
                                let at = start + i * 2;
                                let cur = u16::from_le_bytes([
                                    raw[at],
                                    raw[at + 1],
                                ])
                                .wrapping_add(prev);
                                raw[at..at + 2]
                                    .copy_from_slice(&cur.to_le_bytes());
                                prev = cur;
                            }
                        }
                        SampleType::I32 => {
                            let mut prev = u32::from_le_bytes(
                                raw[start..start + 4].try_into().unwrap(),
                            );
                            for i in 1..row_samples {
                                let at = start + i * 4;
                                let cur = u32::from_le_bytes(
                                    raw[at..at + 4].try_into().unwrap(),
                                )
                                .wrapping_add(prev);
                                raw[at..at + 4]
                                    .copy_from_slice(&cur.to_le_bytes());
                                prev = cur;
                            }
                        }
                        SampleType::F32 | SampleType::F64 => {}
                    }
                }
            }
            3 => {
                // Floating-point predictor: per row, cumulative byte
                // deltas, then byte planes (stored MSB-first)
                // reassembled into the file's declared byte order.
                let le = self.bytes.little_endian;
                let mut assembled = vec![0u8; row_bytes];
                for row in 0..rows {
                    let start = row * row_bytes;
                    let slice = &mut raw[start..start + row_bytes];
                    for i in 1..row_bytes {
                        slice[i] = slice[i].wrapping_add(slice[i - 1]);
                    }
                    for sample_index in 0..row_samples {
                        for byte_index in 0..sample_bytes {
                            let at = if le {
                                sample_bytes - 1 - byte_index
                            } else {
                                byte_index
                            };
                            assembled[sample_index * sample_bytes + at] =
                                slice[byte_index * row_samples
                                    + sample_index];
                        }
                    }
                    slice.copy_from_slice(&assembled);
                }
            }
            _ => {}
        }
    }

    fn sample_to_f64(&self, raw: &[u8], index: usize) -> f64 {
        let s = self.sample.bytes();
        let at = index * s;
        let le = self.bytes.little_endian;
        match self.sample {
            SampleType::U8 => raw[at] as f64,
            SampleType::U16 => {
                let pair = [raw[at], raw[at + 1]];
                (if le { u16::from_le_bytes(pair) } else { u16::from_be_bytes(pair) })
                    as f64
            }
            SampleType::I16 => {
                let pair = [raw[at], raw[at + 1]];
                (if le { i16::from_le_bytes(pair) } else { i16::from_be_bytes(pair) })
                    as f64
            }
            SampleType::I32 => {
                let quad: [u8; 4] = raw[at..at + 4].try_into().unwrap();
                (if le { i32::from_le_bytes(quad) } else { i32::from_be_bytes(quad) })
                    as f64
            }
            SampleType::F32 => {
                let quad: [u8; 4] = raw[at..at + 4].try_into().unwrap();
                (if le { f32::from_le_bytes(quad) } else { f32::from_be_bytes(quad) })
                    as f64
            }
            SampleType::F64 => {
                let oct: [u8; 8] = raw[at..at + 8].try_into().unwrap();
                if le {
                    f64::from_le_bytes(oct)
                } else {
                    f64::from_be_bytes(oct)
                }
            }
        }
    }

    /// Read a window `(col_off, row_off, width, height)` of band 1 as
    /// raw f64 (no masking, no scaling).  The window must lie inside
    /// the image.
    pub fn read_window_raw(
        &mut self,
        col_off: usize,
        row_off: usize,
        win_w: usize,
        win_h: usize,
    ) -> Result<Vec<f64>> {
        if col_off + win_w > self.width || row_off + win_h > self.height {
            return Err(invalid(format!(
                "{:?}: window {col_off},{row_off} {win_w}x{win_h} \
                 leaves the {}x{} image",
                self.path, self.width, self.height
            )));
        }
        let mut out = vec![f64::NAN; win_w * win_h];
        let blocks_across = if self.tiled {
            self.width.div_ceil(self.block_w)
        } else {
            1
        };
        let block_row_lo = row_off / self.block_h;
        let block_row_hi = (row_off + win_h - 1) / self.block_h;
        let block_col_lo = col_off / self.block_w;
        let block_col_hi = (col_off + win_w - 1) / self.block_w;
        for block_row in block_row_lo..=block_row_hi {
            let rows_in_block = if self.tiled {
                self.block_h
            } else {
                (self.height - block_row * self.block_h).min(self.block_h)
            };
            for block_col in block_col_lo..=block_col_hi {
                let index = block_row * blocks_across + block_col;
                if index >= self.offsets.len() {
                    return Err(invalid(format!(
                        "{:?}: block index {index} out of range",
                        self.path
                    )));
                }
                let raw = self.block_bytes(index, rows_in_block)?;
                let base_row = block_row * self.block_h;
                let base_col = block_col * self.block_w;
                let row_lo = row_off.max(base_row);
                let row_hi = (row_off + win_h)
                    .min(base_row + rows_in_block)
                    .min(self.height);
                let col_lo = col_off.max(base_col);
                let col_hi = (col_off + win_w)
                    .min(base_col + self.block_w)
                    .min(self.width);
                for row in row_lo..row_hi {
                    for col in col_lo..col_hi {
                        let value = self.sample_to_f64(
                            &raw,
                            (row - base_row) * self.block_w
                                + (col - base_col),
                        );
                        out[(row - row_off) * win_w + (col - col_off)] =
                            value;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Read a window as a masked, scaled [`Raster`]
    /// (`nodata_override` wins over the file tag; masking on RAW
    /// values before `scale_factor`, matching the Python readers).
    pub fn read_window(
        &mut self,
        col_off: usize,
        row_off: usize,
        win_w: usize,
        win_h: usize,
        crs_override: Option<Crs>,
        nodata_override: Option<f64>,
        scale_factor: f64,
    ) -> Result<Raster> {
        let mut values =
            self.read_window_raw(col_off, row_off, win_w, win_h)?;
        let nodata = nodata_override.or(self.nodata);
        for value in values.iter_mut() {
            let masked = match nodata {
                Some(sentinel) => {
                    *value == sentinel || !value.is_finite()
                }
                None => !value.is_finite(),
            };
            if masked {
                *value = f64::NAN;
            } else {
                *value *= scale_factor;
            }
        }
        let crs = match (&self.crs, crs_override) {
            (Some(own), _) => own.clone(),
            (None, Some(given)) => given,
            (None, None) => {
                return Err(invalid(format!(
                    "raster {:?} has no CRS and declares no crs_override",
                    self.path
                )))
            }
        };
        let t = &self.transform;
        let transform = [
            t[0],
            t[1],
            t[2] + t[0] * col_off as f64 + t[1] * row_off as f64,
            t[3],
            t[4],
            t[5] + t[3] * col_off as f64 + t[4] * row_off as f64,
        ];
        Ok(Raster { ny: win_h, nx: win_w, values, transform, crs })
    }
}

// ---------------------------------------------------------------------------
// GeoKeys
// ---------------------------------------------------------------------------

/// Is `GTRasterTypeGeoKey` (1025) `RasterPixelIsPoint` (2)?
///
/// Absent means `RasterPixelIsArea`, which is the GeoTIFF default and
/// the overwhelmingly common case; only a positive 2 shifts the origin.
/// Read out of the same key directory [`parse_geokeys`] walks, kept
/// separate because the transform is assembled before the CRS is.
fn raster_pixel_is_point(
    tags: &std::collections::BTreeMap<u16, TagEntry>,
    bytes: &ByteReader,
) -> bool {
    let Some(directory) = tags.get(&34735) else {
        return false;
    };
    let shorts: Vec<u16> = directory
        .payload
        .chunks_exact(2)
        .map(|raw| bytes.u16_from(raw))
        .collect();
    if shorts.len() < 4 {
        return false;
    }
    let key_count = shorts[3] as usize;
    for index in 0..key_count {
        let base = 4 + index * 4;
        if base + 3 >= shorts.len() {
            break;
        }
        // key, location, count, value; a short key lives in the
        // directory itself (location 0).
        if shorts[base] == 1025 && shorts[base + 1] == 0 {
            return shorts[base + 3] == 2;
        }
    }
    false
}

fn parse_geokeys(
    path: &Path,
    tags: &std::collections::BTreeMap<u16, TagEntry>,
    bytes: &ByteReader,
) -> Result<Option<Crs>> {
    let Some(directory) = tags.get(&34735) else {
        return Ok(None);
    };
    let shorts: Vec<u16> = directory
        .payload
        .chunks_exact(2)
        .map(|raw| bytes.u16_from(raw))
        .collect();
    if shorts.len() < 4 {
        return Ok(None);
    }
    let doubles: Vec<f64> = tags
        .get(&34736)
        .map(|entry| {
            entry
                .payload
                .chunks_exact(8)
                .map(|raw| bytes.f64_from(raw))
                .collect()
        })
        .unwrap_or_default();

    let key_count = shorts[3] as usize;
    let mut short_keys = std::collections::BTreeMap::new();
    let mut double_keys = std::collections::BTreeMap::new();
    for index in 0..key_count {
        let base = 4 + index * 4;
        if base + 3 >= shorts.len() {
            break;
        }
        let key = shorts[base];
        let location = shorts[base + 1];
        let count = shorts[base + 2] as usize;
        let value = shorts[base + 3];
        match location {
            0 => {
                short_keys.insert(key, value);
            }
            34736 => {
                if count >= 1 && (value as usize) < doubles.len() {
                    double_keys.insert(key, doubles[value as usize]);
                }
            }
            _ => {}
        }
    }

    let model = short_keys.get(&1024).copied().unwrap_or(0);
    match model {
        2 => Ok(Some(Crs::Geographic)),
        1 => {
            let pcs = short_keys.get(&3072).copied().unwrap_or(32767);
            // The one wired projected EPSG code: NAD83 / Conus Albers.
            if pcs == 5070 {
                return Ok(Some(Crs::AlbersConusNad83 {
                    lat_1: 29.5,
                    lat_2: 45.5,
                    lat_0: 23.0,
                    lon_0: -96.0,
                    false_easting: 0.0,
                    false_northing: 0.0,
                }));
            }
            if pcs != 32767 {
                return Err(invalid(format!(
                    "GeoTIFF {path:?}: projected CRS EPSG:{pcs} is \
                     outside the substrate's closed CRS inventory"
                )));
            }
            let method = short_keys.get(&3075).copied().unwrap_or(0);
            if method != 11 {
                return Err(invalid(format!(
                    "GeoTIFF {path:?}: user-defined projection method \
                     {method} is outside the substrate's closed CRS \
                     inventory (Albers equal area = 11)"
                )));
            }
            let get = |candidates: &[u16]| -> Option<f64> {
                candidates
                    .iter()
                    .find_map(|key| double_keys.get(key).copied())
            };
            let lat_1 = get(&[3078]).ok_or_else(|| {
                invalid(format!(
                    "GeoTIFF {path:?}: Albers keys lack StdParallel1"
                ))
            })?;
            let lat_2 = get(&[3079]).ok_or_else(|| {
                invalid(format!(
                    "GeoTIFF {path:?}: Albers keys lack StdParallel2"
                ))
            })?;
            let lat_0 = get(&[3081, 3085, 3089]).ok_or_else(|| {
                invalid(format!(
                    "GeoTIFF {path:?}: Albers keys lack an origin \
                     latitude"
                ))
            })?;
            let lon_0 = get(&[3080, 3084, 3088]).ok_or_else(|| {
                invalid(format!(
                    "GeoTIFF {path:?}: Albers keys lack an origin \
                     longitude"
                ))
            })?;
            let false_easting = get(&[3082, 3086]).unwrap_or(0.0);
            let false_northing = get(&[3083, 3087]).unwrap_or(0.0);
            Ok(Some(Crs::AlbersConusNad83 {
                lat_1,
                lat_2,
                lat_0,
                lon_0,
                false_easting,
                false_northing,
            }))
        }
        0 => Ok(None),
        other => Err(invalid(format!(
            "GeoTIFF {path:?}: GTModelType {other} is outside the \
             substrate's closed CRS inventory"
        ))),
    }
}

/// Read band 1 whole, with masking applied (`nodata_override` wins over
/// the file tag; masking on RAW values before `scale_factor`).
pub fn read_band1(
    path: &Path,
    crs_override: Option<Crs>,
    nodata_override: Option<f64>,
    scale_factor: f64,
) -> Result<Raster> {
    let mut reader = TiffReader::open(path)?;
    let (w, h) = (reader.width, reader.height);
    reader.read_window(0, 0, w, h, crs_override, nodata_override, scale_factor)
}

/// Read band 1 whole, RAW (no masking, no scaling), returning the
/// raster plus the effective nodata (`nodata_override` wins over the
/// file tag) — the mapped-category path reads this way, exactly as
/// the Python reads `dataset.read(1)` next to `dataset.nodata`.
pub fn read_band1_raw(
    path: &Path,
    crs_override: Option<Crs>,
    nodata_override: Option<f64>,
) -> Result<(Raster, Option<f64>)> {
    let mut reader = TiffReader::open(path)?;
    let (w, h) = (reader.width, reader.height);
    let values = reader.read_window_raw(0, 0, w, h)?;
    let nodata = nodata_override.or(reader.nodata);
    let crs = match (&reader.crs, crs_override) {
        (Some(own), _) => own.clone(),
        (None, Some(given)) => given,
        (None, None) => {
            return Err(invalid(format!(
                "raster {path:?} has no CRS and declares no crs_override"
            )))
        }
    };
    Ok((
        Raster { ny: h, nx: w, values, transform: reader.transform, crs },
        nodata,
    ))
}

// ---------------------------------------------------------------------------
// The writer
// ---------------------------------------------------------------------------

const WRITE_TILE: usize = 256;

fn cast_sample(value: f64, sample: SampleType, out: &mut Vec<u8>) {
    match sample {
        SampleType::U8 => out.push(value as u8),
        SampleType::U16 => out.extend((value as u16).to_le_bytes()),
        SampleType::I16 => out.extend((value as i16).to_le_bytes()),
        SampleType::I32 => out.extend((value as i32).to_le_bytes()),
        SampleType::F32 => out.extend((value as f32).to_le_bytes()),
        SampleType::F64 => out.extend(value.to_le_bytes()),
    }
}

fn apply_predictor(
    block: &mut [u8],
    rows: usize,
    row_samples: usize,
    sample: SampleType,
) {
    let sample_bytes = sample.bytes();
    let row_bytes = row_samples * sample_bytes;
    match sample {
        SampleType::F32 | SampleType::F64 => {
            // Predictor 3: byte planes MSB-first, then byte deltas.
            let mut plane = vec![0u8; row_bytes];
            for row in 0..rows {
                let slice = &mut block[row * row_bytes..(row + 1) * row_bytes];
                for sample_index in 0..row_samples {
                    for byte_index in 0..sample_bytes {
                        plane[byte_index * row_samples + sample_index] = slice
                            [sample_index * sample_bytes
                                + (sample_bytes - 1 - byte_index)];
                    }
                }
                for i in (1..row_bytes).rev() {
                    plane[i] = plane[i].wrapping_sub(plane[i - 1]);
                }
                slice.copy_from_slice(&plane);
            }
        }
        SampleType::U8 => {
            for row in 0..rows {
                let start = row * row_bytes;
                for i in (1..row_samples).rev() {
                    block[start + i] =
                        block[start + i].wrapping_sub(block[start + i - 1]);
                }
            }
        }
        SampleType::U16 | SampleType::I16 => {
            for row in 0..rows {
                let start = row * row_bytes;
                for i in (1..row_samples).rev() {
                    let at = start + i * 2;
                    let prev = u16::from_le_bytes([
                        block[at - 2],
                        block[at - 1],
                    ]);
                    let cur =
                        u16::from_le_bytes([block[at], block[at + 1]])
                            .wrapping_sub(prev);
                    block[at..at + 2].copy_from_slice(&cur.to_le_bytes());
                }
            }
        }
        SampleType::I32 => {
            for row in 0..rows {
                let start = row * row_bytes;
                for i in (1..row_samples).rev() {
                    let at = start + i * 4;
                    let prev = u32::from_le_bytes(
                        block[at - 4..at].try_into().unwrap(),
                    );
                    let cur = u32::from_le_bytes(
                        block[at..at + 4].try_into().unwrap(),
                    )
                    .wrapping_sub(prev);
                    block[at..at + 4].copy_from_slice(&cur.to_le_bytes());
                }
            }
        }
    }
}

struct TagWrite {
    tag: u16,
    field_type: u16,
    count: u32,
    /// Inline payload (<= 4 bytes) or out-of-line data.
    data: Vec<u8>,
}

/// Write one band, deflate-compressed, tiled, predictor 2 (integers) /
/// 3 (floats).  `nodata` writes the GDAL nodata tag.  NaN values are
/// written as the nodata sentinel when one is given, else as NaN
/// (floats) / 0 (integers).
pub fn write_band1(
    path: &Path,
    raster: &Raster,
    sample: SampleType,
    nodata: Option<f64>,
) -> Result<()> {
    let (ny, nx) = (raster.ny, raster.nx);
    if ny == 0 || nx == 0 || raster.values.len() != ny * nx {
        return Err(invalid(format!(
            "cannot write {path:?}: raster is {ny}x{nx} with {} values",
            raster.values.len()
        )));
    }
    let sample_bytes = sample.bytes();
    let tiles_across = nx.div_ceil(WRITE_TILE);
    let tiles_down = ny.div_ceil(WRITE_TILE);

    let mut tile_payloads: Vec<Vec<u8>> =
        Vec::with_capacity(tiles_across * tiles_down);
    for tile_row in 0..tiles_down {
        for tile_col in 0..tiles_across {
            let mut block: Vec<u8> =
                Vec::with_capacity(WRITE_TILE * WRITE_TILE * sample_bytes);
            for row in 0..WRITE_TILE {
                for col in 0..WRITE_TILE {
                    let j = tile_row * WRITE_TILE + row;
                    let i = tile_col * WRITE_TILE + col;
                    let mut value = if j < ny && i < nx {
                        raster.values[j * nx + i]
                    } else {
                        0.0
                    };
                    if value.is_nan() {
                        value = match (nodata, sample) {
                            (Some(sentinel), _) => sentinel,
                            (None, SampleType::F32 | SampleType::F64) => {
                                f64::NAN
                            }
                            (None, _) => 0.0,
                        };
                    }
                    cast_sample(value, sample, &mut block);
                }
            }
            apply_predictor(&mut block, WRITE_TILE, WRITE_TILE, sample);
            tile_payloads.push(
                miniz_oxide::deflate::compress_to_vec_zlib(&block, 6),
            );
        }
    }

    let predictor: u16 = match sample {
        SampleType::F32 | SampleType::F64 => 3,
        _ => 2,
    };
    let sample_format: u16 = match sample {
        SampleType::U8 | SampleType::U16 => 1,
        SampleType::I16 | SampleType::I32 => 2,
        SampleType::F32 | SampleType::F64 => 3,
    };

    // Geo tags.
    let t = &raster.transform;
    if t[1] != 0.0 || t[3] != 0.0 || t[0] <= 0.0 || t[4] >= 0.0 {
        return Err(invalid(format!(
            "cannot write {path:?}: transform {t:?} is not north-up \
             rectilinear"
        )));
    }
    let pixel_scale = [t[0], -t[4], 0.0];
    let tiepoint = [0.0, 0.0, 0.0, t[2], t[5], 0.0];
    let mut geo_shorts: Vec<u16> = Vec::new();
    let mut geo_doubles: Vec<f64> = Vec::new();
    match &raster.crs {
        Crs::Geographic => {
            geo_shorts.extend([1, 1, 0, 3]);
            geo_shorts.extend([1024, 0, 1, 2]); // geographic
            geo_shorts.extend([1025, 0, 1, 1]); // RasterPixelIsArea
            geo_shorts.extend([2048, 0, 1, 4326]);
        }
        Crs::InterruptedGoodeHomolosine => {
            // The soil source's own convention: no CRS tag in the file;
            // the override travels in the sidecar/receipt.
        }
        Crs::AlbersConusNad83 {
            lat_1,
            lat_2,
            lat_0,
            lon_0,
            false_easting,
            false_northing,
        } => {
            let mut push_double = |key: u16, value: f64| {
                let index = geo_doubles.len() as u16;
                geo_doubles.push(value);
                [key, 34736, 1, index]
            };
            let d1 = push_double(3078, *lat_1);
            let d2 = push_double(3079, *lat_2);
            let d3 = push_double(3081, *lat_0);
            let d4 = push_double(3080, *lon_0);
            let d5 = push_double(3082, *false_easting);
            let d6 = push_double(3083, *false_northing);
            geo_shorts.extend([1, 1, 0, 10]);
            geo_shorts.extend([1024, 0, 1, 1]); // projected
            geo_shorts.extend([1025, 0, 1, 1]);
            geo_shorts.extend([2048, 0, 1, 4269]); // NAD83 geographic
            geo_shorts.extend([3072, 0, 1, 32767]);
            geo_shorts.extend([3075, 0, 1, 11]); // Albers
            geo_shorts.extend(d1);
            geo_shorts.extend(d2);
            geo_shorts.extend(d3);
            geo_shorts.extend(d4);
            geo_shorts.extend(d5);
            geo_shorts.extend(d6);
        }
        Crs::ModelSphere(_) => {
            return Err(invalid(format!(
                "cannot write {path:?}: model-grid CRS rasters are \
                 in-memory planes, never files"
            )));
        }
    }

    // Assemble tags in ascending order (TIFF requirement).
    let le16 = |values: &[u16]| -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    let le32 = |values: &[u32]| -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    let le64f = |values: &[f64]| -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    };

    let tile_count = tile_payloads.len();
    let mut tags: Vec<TagWrite> = vec![
        TagWrite { tag: 256, field_type: 3, count: 1, data: le16(&[nx as u16]) },
        TagWrite { tag: 257, field_type: 3, count: 1, data: le16(&[ny as u16]) },
        TagWrite {
            tag: 258,
            field_type: 3,
            count: 1,
            data: le16(&[(sample_bytes * 8) as u16]),
        },
        TagWrite { tag: 259, field_type: 3, count: 1, data: le16(&[8]) },
        TagWrite { tag: 262, field_type: 3, count: 1, data: le16(&[1]) },
        TagWrite { tag: 277, field_type: 3, count: 1, data: le16(&[1]) },
        TagWrite { tag: 284, field_type: 3, count: 1, data: le16(&[1]) },
        TagWrite {
            tag: 317,
            field_type: 3,
            count: 1,
            data: le16(&[predictor]),
        },
        TagWrite {
            tag: 322,
            field_type: 3,
            count: 1,
            data: le16(&[WRITE_TILE as u16]),
        },
        TagWrite {
            tag: 323,
            field_type: 3,
            count: 1,
            data: le16(&[WRITE_TILE as u16]),
        },
        // 324/325 filled below once offsets are known.
        TagWrite {
            tag: 339,
            field_type: 3,
            count: 1,
            data: le16(&[sample_format]),
        },
        TagWrite {
            tag: 33550,
            field_type: 12,
            count: 3,
            data: le64f(&pixel_scale),
        },
        TagWrite {
            tag: 33922,
            field_type: 12,
            count: 6,
            data: le64f(&tiepoint),
        },
    ];
    if nx > u16::MAX as usize || ny > u16::MAX as usize {
        // LONG spellings for large derived windows.
        tags[0] = TagWrite {
            tag: 256,
            field_type: 4,
            count: 1,
            data: le32(&[nx as u32]),
        };
        tags[1] = TagWrite {
            tag: 257,
            field_type: 4,
            count: 1,
            data: le32(&[ny as u32]),
        };
    }
    if !geo_shorts.is_empty() {
        tags.push(TagWrite {
            tag: 34735,
            field_type: 3,
            count: geo_shorts.len() as u32,
            data: le16(&geo_shorts),
        });
        if !geo_doubles.is_empty() {
            tags.push(TagWrite {
                tag: 34736,
                field_type: 12,
                count: geo_doubles.len() as u32,
                data: le64f(&geo_doubles),
            });
        }
    }
    if let Some(sentinel) = nodata {
        let mut text = if sentinel.is_nan() {
            "nan".to_string()
        } else {
            format!("{sentinel}")
        };
        text.push('\0');
        tags.push(TagWrite {
            tag: 42113,
            field_type: 2,
            count: text.len() as u32,
            data: text.into_bytes(),
        });
    }

    // Layout: header(8) + IFD + out-of-line tag data + tile data.
    let mut tag_list = tags;
    tag_list.push(TagWrite {
        tag: 324,
        field_type: 4,
        count: tile_count as u32,
        data: vec![0; 4 * tile_count],
    });
    tag_list.push(TagWrite {
        tag: 325,
        field_type: 4,
        count: tile_count as u32,
        data: le32(
            &tile_payloads
                .iter()
                .map(|p| p.len() as u32)
                .collect::<Vec<_>>(),
        ),
    });
    tag_list.sort_by_key(|t| t.tag);

    let ifd_offset = 8u32;
    let entry_count = tag_list.len();
    let ifd_size = 2 + entry_count * 12 + 4;
    let mut extra_offset = ifd_offset as usize + ifd_size;
    let mut extra: Vec<u8> = Vec::new();
    let mut entries: Vec<u8> = Vec::with_capacity(entry_count * 12);

    // First pass to place out-of-line payloads (tile offsets resolved
    // after data placement, so compute the data start now).
    let mut out_of_line_total = 0usize;
    for tag in &tag_list {
        if tag.data.len() > 4 {
            out_of_line_total += tag.data.len() + (tag.data.len() & 1);
        }
    }
    let data_start = extra_offset + out_of_line_total;
    let mut tile_offsets: Vec<u32> = Vec::with_capacity(tile_count);
    let mut cursor = data_start;
    for payload in &tile_payloads {
        tile_offsets.push(cursor as u32);
        cursor += payload.len() + (payload.len() & 1);
    }

    for tag in &mut tag_list {
        if tag.tag == 324 {
            tag.data = le32(&tile_offsets);
        }
    }

    for tag in &tag_list {
        entries.extend(tag.tag.to_le_bytes());
        entries.extend(tag.field_type.to_le_bytes());
        entries.extend(tag.count.to_le_bytes());
        if tag.data.len() <= 4 {
            let mut inline = [0u8; 4];
            inline[..tag.data.len()].copy_from_slice(&tag.data);
            entries.extend(inline);
        } else {
            entries.extend((extra_offset as u32).to_le_bytes());
            extra.extend(&tag.data);
            if tag.data.len() & 1 == 1 {
                extra.push(0);
            }
            extra_offset += tag.data.len() + (tag.data.len() & 1);
        }
    }

    let mut file: Vec<u8> = Vec::with_capacity(cursor);
    file.extend(b"II");
    file.extend(42u16.to_le_bytes());
    file.extend(ifd_offset.to_le_bytes());
    file.extend((entry_count as u16).to_le_bytes());
    file.extend(&entries);
    file.extend(0u32.to_le_bytes()); // next IFD
    file.extend(&extra);
    for payload in &tile_payloads {
        file.extend(payload);
        if payload.len() & 1 == 1 {
            file.push(0);
        }
    }

    let mut handle = File::create(path)?;
    handle.write_all(&file)?;
    handle.sync_all()?;
    Ok(())
}
