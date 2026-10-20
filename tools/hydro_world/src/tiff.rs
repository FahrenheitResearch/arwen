//! Minimal GeoTIFF reader for the builder's raster inputs (DEM tiles).
//! Classic TIFF (II/MM), tiled or stripped, one sample per pixel, compression
//! none or Deflate (8 / 32946), predictor 1, 2 (integer horizontal) or 3
//! (floating point, libtiff fpAcc byte-plane layout), sample types float32,
//! int16, uint16, uint8. Georeferencing from ModelTiepoint + ModelPixelScale
//! and GTRasterTypeGeoKey (PixelIsArea / PixelIsPoint). Anything outside this
//! set is refused by name rather than decoded wrongly.
use crate::Result;
use std::io::Read;

#[derive(Debug, Clone)]
pub struct Raster {
    pub width: usize,
    pub height: usize,
    /// Row-major, north row first, values as f32.
    pub data: Vec<f32>,
    /// Longitude/x of the centre of column 0.
    pub x0: f64,
    /// Latitude/y of the centre of row 0.
    pub y0: f64,
    /// Column spacing (positive).
    pub dx: f64,
    /// Row spacing (positive; rows run southwards).
    pub dy: f64,
    pub nodata: Option<f32>,
}

struct Ifd {
    le: bool,
    entries: Vec<(u16, u16, u32, [u8; 4])>,
}

fn rd16(b: &[u8], o: usize, le: bool) -> Result<u16> {
    let s: [u8; 2] = b.get(o..o + 2).ok_or("tiff truncated")?.try_into()?;
    Ok(if le { u16::from_le_bytes(s) } else { u16::from_be_bytes(s) })
}
fn rd32(b: &[u8], o: usize, le: bool) -> Result<u32> {
    let s: [u8; 4] = b.get(o..o + 4).ok_or("tiff truncated")?.try_into()?;
    Ok(if le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) })
}
fn rd64f(b: &[u8], o: usize, le: bool) -> Result<f64> {
    let s: [u8; 8] = b.get(o..o + 8).ok_or("tiff truncated")?.try_into()?;
    Ok(if le { f64::from_le_bytes(s) } else { f64::from_be_bytes(s) })
}

impl Ifd {
    fn find(&self, tag: u16) -> Option<&(u16, u16, u32, [u8; 4])> {
        self.entries.iter().find(|e| e.0 == tag)
    }
    fn type_size(t: u16) -> usize {
        match t {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 => 4,
            5 | 10 | 12 => 8,
            _ => 0,
        }
    }
    fn data_offset(&self, e: &(u16, u16, u32, [u8; 4])) -> Option<usize> {
        let n = Self::type_size(e.1) * e.2 as usize;
        if n <= 4 {
            None
        } else {
            let o = if self.le { u32::from_le_bytes(e.3) } else { u32::from_be_bytes(e.3) };
            Some(o as usize)
        }
    }
    fn uints(&self, b: &[u8], tag: u16) -> Result<Vec<u64>> {
        let e = self.find(tag).ok_or_else(|| format!("tiff tag {tag} missing"))?;
        let n = e.2 as usize;
        let mut v = Vec::with_capacity(n);
        let off = self.data_offset(e);
        for i in 0..n {
            let x = match (e.1, off) {
                (3, None) => rd16(&e.3, 2 * i, self.le)? as u64,
                (3, Some(o)) => rd16(b, o + 2 * i, self.le)? as u64,
                (4, None) => rd32(&e.3, 0, self.le)? as u64,
                (4, Some(o)) => rd32(b, o + 4 * i, self.le)? as u64,
                (1, None) => e.3[i] as u64,
                (1, Some(o)) => *b.get(o + i).ok_or("tiff truncated")? as u64,
                (t, _) => return Err(format!("tiff tag {tag} type {t} is not an unsigned integer type").into()),
            };
            v.push(x);
        }
        Ok(v)
    }
    fn uint(&self, b: &[u8], tag: u16) -> Result<u64> {
        Ok(self.uints(b, tag)?[0])
    }
    fn uint_or(&self, b: &[u8], tag: u16, d: u64) -> Result<u64> {
        if self.find(tag).is_some() { self.uint(b, tag) } else { Ok(d) }
    }
    fn doubles(&self, b: &[u8], tag: u16) -> Result<Vec<f64>> {
        let e = self.find(tag).ok_or_else(|| format!("tiff tag {tag} missing"))?;
        if e.1 != 12 {
            return Err(format!("tiff tag {tag} is not DOUBLE").into());
        }
        let o = self.data_offset(e).ok_or("tiff double inline")?;
        (0..e.2 as usize).map(|i| rd64f(b, o + 8 * i, self.le)).collect()
    }
    fn ascii(&self, b: &[u8], tag: u16) -> Option<String> {
        let e = self.find(tag)?;
        if e.1 != 2 {
            return None;
        }
        let n = e.2 as usize;
        let bytes = match self.data_offset(e) {
            None => e.3[..n.min(4)].to_vec(),
            Some(o) => b.get(o..o + n)?.to_vec(),
        };
        Some(String::from_utf8_lossy(&bytes).trim_end_matches('\0').trim().to_string())
    }
}

/// Decode a whole GeoTIFF file held in memory.
pub fn decode(b: &[u8]) -> Result<Raster> {
    let le = match b.get(..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return Err("not a TIFF".into()),
    };
    if rd16(b, 2, le)? != 42 {
        return Err("BigTIFF or non-TIFF: only classic TIFF is supported".into());
    }
    let off = rd32(b, 4, le)? as usize;
    let n = rd16(b, off, le)? as usize;
    let mut entries = Vec::with_capacity(n);
    for i in 0..n {
        let o = off + 2 + 12 * i;
        let raw: [u8; 4] = b.get(o + 8..o + 12).ok_or("tiff truncated")?.try_into()?;
        entries.push((rd16(b, o, le)?, rd16(b, o + 2, le)?, rd32(b, o + 4, le)?, raw));
    }
    let ifd = Ifd { le, entries };
    let width = ifd.uint(b, 256)? as usize;
    let height = ifd.uint(b, 257)? as usize;
    let bps = ifd.uint_or(b, 258, 1)? as usize;
    let comp = ifd.uint_or(b, 259, 1)?;
    let spp = ifd.uint_or(b, 277, 1)?;
    let planar = ifd.uint_or(b, 284, 1)?;
    let pred = ifd.uint_or(b, 317, 1)?;
    let fmt = ifd.uint_or(b, 339, 1)?;
    if spp != 1 || planar != 1 {
        return Err(format!("tiff samples_per_pixel={spp} planar={planar}: only single-band rasters are supported").into());
    }
    if !matches!(comp, 1 | 8 | 32946) {
        return Err(format!("tiff compression {comp} is not supported (none or deflate)").into());
    }
    let kind = match (fmt, bps) {
        (3, 32) => 'f',
        (2, 16) => 'i',
        (1, 16) => 'u',
        (1, 8) => 'b',
        _ => return Err(format!("tiff sample format {fmt} with {bps} bits is not supported").into()),
    };
    let bytes_per = bps / 8;
    let (bw, bh, offsets, counts) = if ifd.find(322).is_some() {
        (ifd.uint(b, 322)? as usize, ifd.uint(b, 323)? as usize, ifd.uints(b, 324)?, ifd.uints(b, 325)?)
    } else {
        let rps = (ifd.uint_or(b, 278, height as u64)? as usize).min(height);
        (width, rps, ifd.uints(b, 273)?, ifd.uints(b, 279)?)
    };
    let across = width.div_ceil(bw);
    let down = height.div_ceil(bh);
    if offsets.len() < across * down || counts.len() < offsets.len() {
        return Err("tiff block table shorter than the image".into());
    }
    let mut data = vec![0f32; width * height];
    let row_bytes = bw * bytes_per;
    let mut block = Vec::new();
    for by in 0..down {
        for bx in 0..across {
            let k = by * across + bx;
            let (o, c) = (offsets[k] as usize, counts[k] as usize);
            let raw = b.get(o..o + c).ok_or("tiff block out of range")?;
            block.clear();
            if comp == 1 {
                block.extend_from_slice(raw);
            } else {
                flate2::read::ZlibDecoder::new(raw).read_to_end(&mut block)?;
            }
            let rows_here = if ifd.find(322).is_some() { bh } else { bh.min(height - by * bh) };
            if block.len() < rows_here * row_bytes {
                return Err("tiff block decompressed short".into());
            }
            let mut tmp = vec![0u8; row_bytes];
            for r in 0..rows_here {
                let row = &mut block[r * row_bytes..(r + 1) * row_bytes];
                match pred {
                    1 => {}
                    2 => {
                        // Horizontal differencing on whole samples.
                        match bytes_per {
                            1 => {
                                for i in 1..bw {
                                    row[i] = row[i].wrapping_add(row[i - 1]);
                                }
                            }
                            2 => {
                                for i in 1..bw {
                                    let p = if le { u16::from_le_bytes([row[2 * i - 2], row[2 * i - 1]]) } else { u16::from_be_bytes([row[2 * i - 2], row[2 * i - 1]]) };
                                    let c = if le { u16::from_le_bytes([row[2 * i], row[2 * i + 1]]) } else { u16::from_be_bytes([row[2 * i], row[2 * i + 1]]) };
                                    let v = c.wrapping_add(p);
                                    let bb = if le { v.to_le_bytes() } else { v.to_be_bytes() };
                                    row[2 * i] = bb[0];
                                    row[2 * i + 1] = bb[1];
                                }
                            }
                            _ => return Err("tiff predictor 2 on 32-bit samples is not supported".into()),
                        }
                    }
                    3 => {
                        // libtiff fpAcc: byte-wise accumulation, then byte planes
                        // (most significant first) re-interleaved.
                        for i in 1..row_bytes {
                            row[i] = row[i].wrapping_add(row[i - 1]);
                        }
                        tmp.copy_from_slice(row);
                        for j in 0..bw {
                            for p in 0..bytes_per {
                                // big-endian assembled value
                                row[j * bytes_per + p] = tmp[p * bw + j];
                            }
                        }
                    }
                    p => return Err(format!("tiff predictor {p} is not supported").into()),
                }
                let y = by * bh + r;
                if y >= height {
                    break;
                }
                let be_assembled = pred == 3;
                for j in 0..bw {
                    let x = bx * bw + j;
                    if x >= width {
                        break;
                    }
                    let s = &row[j * bytes_per..(j + 1) * bytes_per];
                    let v = match kind {
                        'f' => {
                            let a: [u8; 4] = s.try_into()?;
                            if be_assembled || !le { f32::from_be_bytes(a) } else { f32::from_le_bytes(a) }
                        }
                        'i' => {
                            let a: [u8; 2] = s.try_into()?;
                            (if be_assembled || !le { i16::from_be_bytes(a) } else { i16::from_le_bytes(a) }) as f32
                        }
                        'u' => {
                            let a: [u8; 2] = s.try_into()?;
                            (if be_assembled || !le { u16::from_be_bytes(a) } else { u16::from_le_bytes(a) }) as f32
                        }
                        _ => s[0] as f32,
                    };
                    data[y * width + x] = v;
                }
            }
        }
    }
    let scale = ifd.doubles(b, 33550)?;
    let tie = ifd.doubles(b, 33922)?;
    if tie.len() < 6 || scale.len() < 2 {
        return Err("tiff georeferencing tags incomplete".into());
    }
    // GTRasterTypeGeoKey (1025): 1 = PixelIsArea (tiepoint at a pixel corner), 2 = PixelIsPoint.
    let mut point = false;
    if let Ok(keys) = ifd.uints(b, 34735) {
        for k in keys.chunks(4).skip(1) {
            if k.len() == 4 && k[0] == 1025 && k[1] == 0 {
                point = k[3] == 2;
            }
        }
    }
    let half = if point { 0.0 } else { 0.5 };
    let (dx, dy) = (scale[0], scale[1]);
    let x0 = tie[3] + (0.0 - tie[0] + half) * dx;
    let y0 = tie[4] - (0.0 - tie[1] + half) * dy;
    let nodata = ifd.ascii(b, 42113).and_then(|s| s.parse::<f32>().ok());
    Ok(Raster { width, height, data, x0, y0, dx, dy, nodata })
}

pub fn read(path: &std::path::Path) -> Result<Raster> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode(&b).map_err(|e| format!("{}: {e}", path.display()).into())
}
