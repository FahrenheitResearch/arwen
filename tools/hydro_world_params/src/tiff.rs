//! Minimal GeoTIFF reader for the hydro-world parameter inputs (Copernicus DEM
//! COGs, ESA WorldCover COGs, SoilGrids WCS GeoTIFFs). Reads the first IFD
//! (full resolution) only. Supports classic TIFF and BigTIFF, little or big
//! endian, tiled or stripped, no compression / LZW / Deflate, predictor 1, 2
//! (horizontal differencing) and 3 (floating point, Adobe TIFF Technical Note 3).
//! Blocks decode independently, so callers decode them in parallel with Rayon.
use crate::Result;
use std::io::Read;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampleKind {
    U8,
    I16,
    U16,
    I32,
    F32,
}

impl SampleKind {
    pub fn bytes(self) -> usize {
        match self {
            SampleKind::U8 => 1,
            SampleKind::I16 | SampleKind::U16 => 2,
            SampleKind::I32 | SampleKind::F32 => 4,
        }
    }
}

pub struct GeoTiff {
    pub path: String,
    bytes: Vec<u8>,
    le: bool,
    pub width: usize,
    pub height: usize,
    pub kind: SampleKind,
    compression: u16,
    predictor: u16,
    /// block size; a strip is a block of width x rows_per_strip
    pub block_w: usize,
    pub block_h: usize,
    tiled: bool,
    offsets: Vec<u64>,
    counts: Vec<u64>,
    /// west edge of column 0 and north edge of row 0 (pixel-is-area), degrees
    pub origin_x: f64,
    pub origin_y: f64,
    /// pixel size, both positive (rows run north to south)
    pub dx: f64,
    pub dy: f64,
    pub nodata: Option<f64>,
}

fn rd_u16(b: &[u8], o: usize, le: bool) -> u16 {
    let a = [b[o], b[o + 1]];
    if le { u16::from_le_bytes(a) } else { u16::from_be_bytes(a) }
}
fn rd_u32(b: &[u8], o: usize, le: bool) -> u32 {
    let a = [b[o], b[o + 1], b[o + 2], b[o + 3]];
    if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) }
}
fn rd_u64(b: &[u8], o: usize, le: bool) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    if le { u64::from_le_bytes(a) } else { u64::from_be_bytes(a) }
}
fn rd_f64(b: &[u8], o: usize, le: bool) -> f64 {
    f64::from_bits(rd_u64(b, o, le))
}

struct Entry {
    typ: u16,
    count: u64,
    /// absolute offset of the value bytes
    at: usize,
}

impl GeoTiff {
    pub fn open(path: &str) -> Result<GeoTiff> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Self::from_bytes(path, bytes)
    }

    pub fn from_bytes(path: &str, bytes: Vec<u8>) -> Result<GeoTiff> {
        if bytes.len() < 16 {
            return Err(format!("{path}: too short for TIFF").into());
        }
        let le = match &bytes[0..2] {
            b"II" => true,
            b"MM" => false,
            _ => return Err(format!("{path}: not a TIFF").into()),
        };
        let magic = rd_u16(&bytes, 2, le);
        let big = match magic {
            42 => false,
            43 => true,
            _ => return Err(format!("{path}: bad TIFF magic {magic}").into()),
        };
        let ifd = if big { rd_u64(&bytes, 8, le) as usize } else { rd_u32(&bytes, 4, le) as usize };
        let (n, esz, first) = if big {
            (rd_u64(&bytes, ifd, le) as usize, 20usize, ifd + 8)
        } else {
            (rd_u16(&bytes, ifd, le) as usize, 12usize, ifd + 2)
        };
        let mut entries: std::collections::BTreeMap<u16, Entry> = Default::default();
        for i in 0..n {
            let e = first + i * esz;
            let tag = rd_u16(&bytes, e, le);
            let typ = rd_u16(&bytes, e + 2, le);
            let (count, voff, inline_len) = if big {
                (rd_u64(&bytes, e + 4, le), e + 12, 8usize)
            } else {
                (rd_u32(&bytes, e + 4, le) as u64, e + 8, 4usize)
            };
            let tsz = match typ {
                1 | 2 | 6 | 7 => 1,
                3 | 8 => 2,
                4 | 9 | 11 => 4,
                5 | 10 | 12 | 16 | 17 | 18 => 8,
                _ => 1,
            } as u64;
            let at = if tsz * count <= inline_len as u64 {
                voff
            } else if big {
                rd_u64(&bytes, voff, le) as usize
            } else {
                rd_u32(&bytes, voff, le) as usize
            };
            entries.insert(tag, Entry { typ, count, at });
        }
        let nums = |tag: u16| -> Option<Vec<f64>> {
            let e = entries.get(&tag)?;
            let mut v = Vec::with_capacity(e.count as usize);
            for k in 0..e.count as usize {
                let x = match e.typ {
                    1 | 7 => bytes[e.at + k] as f64,
                    6 => bytes[e.at + k] as i8 as f64,
                    3 => rd_u16(&bytes, e.at + 2 * k, le) as f64,
                    8 => rd_u16(&bytes, e.at + 2 * k, le) as i16 as f64,
                    4 => rd_u32(&bytes, e.at + 4 * k, le) as f64,
                    9 => rd_u32(&bytes, e.at + 4 * k, le) as i32 as f64,
                    11 => f32::from_bits(rd_u32(&bytes, e.at + 4 * k, le)) as f64,
                    12 => rd_f64(&bytes, e.at + 8 * k, le),
                    16 => rd_u64(&bytes, e.at + 8 * k, le) as f64,
                    5 => rd_u32(&bytes, e.at + 8 * k, le) as f64 / rd_u32(&bytes, e.at + 8 * k + 4, le) as f64,
                    _ => return None,
                };
                v.push(x);
            }
            Some(v)
        };
        let ints = |tag: u16| -> Option<Vec<u64>> {
            let e = entries.get(&tag)?;
            let mut v = Vec::with_capacity(e.count as usize);
            for k in 0..e.count as usize {
                v.push(match e.typ {
                    3 => rd_u16(&bytes, e.at + 2 * k, le) as u64,
                    4 => rd_u32(&bytes, e.at + 4 * k, le) as u64,
                    16 => rd_u64(&bytes, e.at + 8 * k, le),
                    _ => return None,
                });
            }
            Some(v)
        };
        let one = |tag: u16, dflt: Option<f64>| -> Result<f64> {
            nums(tag).and_then(|v| v.first().copied()).or(dflt).ok_or_else(|| format!("{path}: missing TIFF tag {tag}").into())
        };
        let width = one(256, None)? as usize;
        let height = one(257, None)? as usize;
        let bps = one(258, Some(1.0))? as u32;
        let spp = one(277, Some(1.0))? as u32;
        if spp != 1 {
            return Err(format!("{path}: {spp} samples per pixel; only single-band rasters are read").into());
        }
        let fmt = one(339, Some(1.0))? as u32;
        let kind = match (fmt, bps) {
            (1, 8) => SampleKind::U8,
            (2, 16) => SampleKind::I16,
            (1, 16) => SampleKind::U16,
            (2, 32) => SampleKind::I32,
            (3, 32) => SampleKind::F32,
            _ => return Err(format!("{path}: unsupported sample format {fmt}/{bps} bits").into()),
        };
        let compression = one(259, Some(1.0))? as u16;
        if ![1u16, 5, 8, 32946].contains(&compression) {
            return Err(format!("{path}: unsupported compression {compression}").into());
        }
        let predictor = one(317, Some(1.0))? as u16;
        let (tiled, block_w, block_h, offsets, counts) = if entries.contains_key(&322) {
            let tw = one(322, None)? as usize;
            let th = one(323, None)? as usize;
            (true, tw, th, ints(324).ok_or("tile offsets")?, ints(325).ok_or("tile byte counts")?)
        } else {
            let rps = one(278, Some(height as f64))? as usize;
            (false, width, rps.min(height), ints(273).ok_or("strip offsets")?, ints(279).ok_or("strip byte counts")?)
        };
        let scale = nums(33550).ok_or_else(|| format!("{path}: no ModelPixelScale (not a GeoTIFF)"))?;
        let tie = nums(33922).ok_or_else(|| format!("{path}: no ModelTiepoint (not a GeoTIFF)"))?;
        // GeoKey 1025 RasterPixelIsArea(1)/Point(2); the four inputs are all
        // PixelIsArea (GDAL reports AREA_OR_POINT=Area), so the tiepoint is the
        // corner of pixel (0,0).
        let mut origin_x = tie[3] - tie[0] * scale[0];
        let mut origin_y = tie[4] + tie[1] * scale[1];
        if let Some(gk) = nums(34735) {
            let nk = gk.get(3).copied().unwrap_or(0.0) as usize;
            for k in 0..nk {
                let b = 4 + 4 * k;
                if gk.len() > b + 3 && gk[b] as u32 == 1025 && gk[b + 1] as u32 == 0 && gk[b + 3] as u32 == 2 {
                    // PixelIsPoint: tiepoint is the centre of pixel (0,0)
                    origin_x -= 0.5 * scale[0];
                    origin_y += 0.5 * scale[1];
                }
            }
        }
        let nodata = entries.get(&42113).and_then(|e| {
            let s = String::from_utf8_lossy(&bytes[e.at..e.at + e.count as usize]).trim_matches(char::from(0)).trim().to_string();
            s.parse::<f64>().ok()
        });
        Ok(GeoTiff {
            path: path.to_string(),
            bytes,
            le,
            width,
            height,
            kind,
            compression,
            predictor,
            block_w,
            block_h,
            tiled,
            offsets,
            counts,
            origin_x,
            origin_y,
            dx: scale[0],
            dy: scale[1],
            nodata,
        })
    }

    pub fn blocks_across(&self) -> usize {
        if self.tiled { self.width.div_ceil(self.block_w) } else { 1 }
    }
    pub fn blocks_down(&self) -> usize {
        self.height.div_ceil(self.block_h)
    }
    pub fn n_blocks(&self) -> usize {
        self.blocks_across() * self.blocks_down()
    }
    /// (x0, y0, valid width, valid height, stored row length) of block `b`
    pub fn block_rect(&self, b: usize) -> (usize, usize, usize, usize, usize) {
        let bx = b % self.blocks_across();
        let by = b / self.blocks_across();
        let x0 = bx * self.block_w;
        let y0 = by * self.block_h;
        let w = self.block_w.min(self.width - x0);
        let h = self.block_h.min(self.height - y0);
        (x0, y0, w, h, self.block_w)
    }

    /// Decompress block `b` and undo the predictor. Returns native-endian
    /// sample bytes, `stored_row_len * rows` samples (tiles are padded).
    pub fn decode_block(&self, b: usize) -> Result<Vec<u8>> {
        let (_, _, _, h, row_len) = self.block_rect(b);
        let rows = if self.tiled { self.block_h } else { h };
        let ss = self.kind.bytes();
        let want = row_len * rows * ss;
        let off = self.offsets[b] as usize;
        let cnt = self.counts[b] as usize;
        let src = self.bytes.get(off..off + cnt).ok_or_else(|| format!("{}: block {b} outside file", self.path))?;
        let mut raw = match self.compression {
            1 => src.to_vec(),
            8 | 32946 => {
                let mut out = Vec::with_capacity(want);
                flate2::read::ZlibDecoder::new(src).read_to_end(&mut out).map_err(|e| format!("{}: deflate block {b}: {e}", self.path))?;
                out
            }
            5 => weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
                .decode(src)
                .map_err(|e| format!("{}: LZW block {b}: {e:?}", self.path))?,
            c => return Err(format!("compression {c}").into()),
        };
        if raw.len() < want {
            raw.resize(want, 0);
        }
        raw.truncate(want);
        match self.predictor {
            1 => {
                // samples are returned little-endian (the only host this runs on)
                if ss > 1 && !self.le {
                    for c in raw.chunks_exact_mut(ss) {
                        c.reverse();
                    }
                }
            }
            2 => {
                // byte order first, then horizontal differencing on samples
                if ss > 1 && !self.le {
                    for c in raw.chunks_exact_mut(ss) {
                        c.reverse();
                    }
                }
                for r in 0..rows {
                    let row = &mut raw[r * row_len * ss..(r + 1) * row_len * ss];
                    match ss {
                        1 => {
                            for i in 1..row_len {
                                row[i] = row[i].wrapping_add(row[i - 1]);
                            }
                        }
                        2 => {
                            for i in 1..row_len {
                                let p = u16::from_le_bytes([row[2 * i - 2], row[2 * i - 1]]);
                                let c = u16::from_le_bytes([row[2 * i], row[2 * i + 1]]);
                                row[2 * i..2 * i + 2].copy_from_slice(&c.wrapping_add(p).to_le_bytes());
                            }
                        }
                        4 => {
                            for i in 1..row_len {
                                let p = u32::from_le_bytes(row[4 * i - 4..4 * i].try_into().unwrap());
                                let c = u32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap());
                                row[4 * i..4 * i + 4].copy_from_slice(&c.wrapping_add(p).to_le_bytes());
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                if !cfg!(target_endian = "little") {
                    return Err("big-endian host not supported".into());
                }
            }
            3 => {
                // Floating point predictor (Adobe Photoshop TIFF Technical Note 3):
                // bytes are differenced across the row, and each row stores the
                // most significant byte of every sample first.
                let mut tmp = vec![0u8; row_len * ss];
                for r in 0..rows {
                    let row = &mut raw[r * row_len * ss..(r + 1) * row_len * ss];
                    for i in 1..row.len() {
                        row[i] = row[i].wrapping_add(row[i - 1]);
                    }
                    for i in 0..row_len {
                        let mut be = [0u8; 8];
                        for k in 0..ss {
                            be[k] = row[k * row_len + i];
                        }
                        // big-endian sample -> native little-endian bytes
                        for k in 0..ss {
                            tmp[i * ss + k] = be[ss - 1 - k];
                        }
                    }
                    row.copy_from_slice(&tmp);
                }
            }
            p => return Err(format!("{}: unsupported predictor {p}", self.path).into()),
        }
        Ok(raw)
    }

    /// Whole raster as f32 (row-major, north row first). Blocks decode in parallel.
    pub fn read_all_f32(&self) -> Result<Vec<f32>> {
        use rayon::prelude::*;
        let mut out = vec![f32::NAN; self.width * self.height];
        let blocks: Vec<Result<(usize, Vec<u8>)>> = (0..self.n_blocks()).into_par_iter().map(|b| self.decode_block(b).map(|d| (b, d))).collect();
        for r in blocks {
            let (b, d) = r?;
            let (x0, y0, w, h, row_len) = self.block_rect(b);
            for r in 0..h {
                for c in 0..w {
                    let i = r * row_len + c;
                    out[(y0 + r) * self.width + x0 + c] = sample_f32(self.kind, &d, i);
                }
            }
        }
        if let Some(nd) = self.nodata {
            let nd = nd as f32;
            for v in out.iter_mut() {
                if *v == nd {
                    *v = f32::NAN;
                }
            }
        }
        Ok(out)
    }
}

#[inline]
pub fn sample_f32(kind: SampleKind, d: &[u8], i: usize) -> f32 {
    match kind {
        SampleKind::U8 => d[i] as f32,
        SampleKind::I16 => i16::from_le_bytes([d[2 * i], d[2 * i + 1]]) as f32,
        SampleKind::U16 => u16::from_le_bytes([d[2 * i], d[2 * i + 1]]) as f32,
        SampleKind::I32 => i32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()) as f32,
        SampleKind::F32 => f32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()),
    }
}
