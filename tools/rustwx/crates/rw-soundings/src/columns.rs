//! The any-point column store (`gpuwm-columns.v1`): every column of a run's grid at one valid time, chunked so one
//! sounding anywhere in the domain is one small read.
//!
//! A forecast box writes one file per domain and valid time from its WRF-shaped history (ArWen's wrfout, hex's
//! lat-lon export). An always-on service later answers "the sounding at this point and time" by reading the file's
//! header once, then the one tile that holds the point (an HTTP range read), sampling the column bilinearly and
//! handing it to `rw_soundings` as a JSON column frame. Nobody has to pick sounding points before a run.
//!
//! Layout, little-endian:
//! ```text
//! 0..8      b"GPWMCOL1"
//! 8..12     u32 header length H
//! 12..12+H  JSON header: schema, domain, member, valid, nx, ny, nz, tile, tiles_x, tiles_y, fields3d, quanta3d,
//!           fields2d, dtype2d (f32), codec (shuffle-zlib), filled, index_offset, locator {step, ix, iy, lat, lon}
//! index     tiles_y * tiles_x entries of (u64 offset from file start, u32 length), row-major (ty outer)
//! tiles     each tile zlib(byte-shuffled payload)
//! ```
//! Tile (tx, ty) holds columns x in [tx*T, min(tx*T+T, nx-1)] and y likewise: tiles overlap by one column, so the
//! four corners of any grid cell lie in one tile and a bilinear sample is one read. Payload: one byte W (2 or 4),
//! then every 3-D field [field][k][y][x] as integer multiples of its quantum (`quanta3d`), differenced up the
//! column and zigzagged, W bytes each, byte-shuffled into W planes; then each 2-D field as f32 [y][x], shuffled into
//! 4 planes; the whole payload zlib-compressed. Units: hPa, m MSL, C, m/s (earth-relative), Pa/s; degrees.
use crate::input::{self, Source, WrfGrid};
use crate::store::Res;
use flate2::{Compression, write::ZlibEncoder};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};
use wrf_core::WrfFile;

pub const MAGIC: &[u8; 8] = b"GPWMCOL1";
pub const FIELDS_3D: [&str; 7] = ["p_hpa", "z_m", "t_c", "td_c", "u_ms", "v_ms", "omega_pas"];
pub const FIELDS_2D: [&str; 8] = ["lat", "lon", "psfc_hpa", "orog_m", "t2_c", "td2_c", "u10_ms", "v10_ms"];

#[derive(Deserialize)]
pub struct ColumnsRequest {
    pub sources: Vec<Source>,
    pub out: PathBuf,
    #[serde(default = "default_tile")]
    pub tile: usize,
    #[serde(default = "default_threads")]
    pub threads: usize,
}
fn default_tile() -> usize {
    16
}
fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn stamp(t: i64) -> String {
    chrono::DateTime::from_timestamp(t, 0).unwrap().format("%Y%m%dT%H%M%SZ").to_string()
}

/// One written file, as the caller's receipt wants it.
pub struct Written {
    pub path: PathBuf,
    pub bytes: u64,
    pub ms: f64,
}

pub fn run(req: ColumnsRequest) -> Res<Vec<Written>> {
    if req.sources.is_empty() {
        return Err("a column store needs at least one history file".into());
    }
    if !(4..=128).contains(&req.tile) {
        return Err(format!("tile must be 4..128 columns, got {}", req.tile));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(req.threads.max(1))
        .build()
        .map_err(err)?;
    let mut written = vec![];
    // Files one at a time (a 3 km nest's time is ~1 GB of f64 fields); the tiles of each in parallel.
    for source in &req.sources {
        let f = WrfFile::open(&source.path).map_err(|e| format!("{}: {e}", source.path.display()))?;
        for (ti, time) in f.times().map_err(err)?.iter().enumerate() {
            let started = std::time::Instant::now();
            f.prepare_cache_for_time(ti);
            let grid = input::wrf_grid(&f, ti)?;
            let valid = input::parse_time(time)?;
            let dir = if source.member == "det" {
                req.out.join(&source.domain)
            } else {
                req.out.join("members").join(&source.member).join(&source.domain)
            };
            std::fs::create_dir_all(&dir).map_err(err)?;
            let path = dir.join(format!("{}.wcol", stamp(valid)));
            let bytes = pool.install(|| write_file(&path, &grid, &source.domain, &source.member, valid, req.tile))?;
            drop(grid);
            f.clear_cache();
            let ms = started.elapsed().as_secs_f64() * 1e3;
            println!("COLUMNS {} bytes={} ms={:.0}", path.display(), bytes, ms);
            written.push(Written { path, bytes, ms });
        }
    }
    Ok(written)
}

fn span(t: usize, tile: usize, n: usize) -> (usize, usize) {
    let a = t * tile;
    (a, (a + tile).min(n - 1))
}

/// Quantum of each 3-D field in its store unit: 0.05 hPa, 0.5 m, 0.02 C, 0.05 C, 0.05 m/s, 0.05 m/s, 0.01 Pa/s.
pub const QUANTA_3D: [f64; 7] = [0.05, 0.5, 0.02, 0.05, 0.05, 0.05, 0.01];

fn encode_tile(g: &WrfGrid, td2: &[f64], x0: usize, x1: usize, y0: usize, y1: usize) -> Res<(Vec<u8>, usize)> {
    let (w, h, nz) = (x1 - x0 + 1, y1 - y0 + 1, g.nz);
    let n = g.nx * g.ny;
    let plane = w * h;
    // 3-D fields: quantized to QUANTA_3D, differenced up the column (a column is smooth in the vertical, so the
    // differences are small), zigzagged to unsigned. (field, factor to the store unit): pressure Pa -> hPa.
    let fields3: [(&[f64], f64); 7] =
        [(&g.p, 0.01), (&g.z, 1.0), (&g.t, 1.0), (&g.td, 1.0), (&g.u, 1.0), (&g.v, 1.0), (&g.omega, 1.0)];
    let mut codes: Vec<u32> = Vec::with_capacity(fields3.len() * nz * plane);
    let mut filled = 0usize;
    let mut last = vec![0i64; plane];
    for ((a, factor), quantum) in fields3.iter().zip(QUANTA_3D) {
        last.iter_mut().for_each(|v| *v = 0);
        for k in 0..nz {
            for y in y0..=y1 {
                let row = k * n + y * g.nx;
                for x in x0..=x1 {
                    let c = (y - y0) * w + (x - x0);
                    let value = a[row + x] * factor / quantum;
                    // A missing value repeats the level below (a zero difference); counted, never silent.
                    let q = if value.is_finite() {
                        value.round() as i64
                    } else {
                        filled += 1;
                        last[c]
                    };
                    let d = q - last[c];
                    last[c] = q;
                    let d = d.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                    codes.push(((d << 1) ^ (d >> 31)) as u32);
                }
            }
        }
    }
    let mut floats: Vec<f32> = Vec::with_capacity(FIELDS_2D.len() * plane);
    // (field, scale, offset): surface pressure Pa -> hPa, 2 m temperature K -> C.
    let fields2: [(&[f64], f64, f64); 8] = [
        (&g.lat, 1.0, 0.0),
        (&g.lon, 1.0, 0.0),
        (&g.ps, 0.01, 0.0),
        (&g.terrain, 1.0, 0.0),
        (&g.t2, 1.0, -273.15),
        (td2, 1.0, 0.0),
        (&g.u10, 1.0, 0.0),
        (&g.v10, 1.0, 0.0),
    ];
    for (a, scale, offset) in fields2 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                floats.push((a[y * g.nx + x] * scale + offset) as f32);
            }
        }
    }
    // Byte shuffle: plane b holds byte b of every value, so the high bytes (mostly zero) compress together.
    // The first byte says how wide the 3-D codes are: 2 bytes when every one fits, else 4.
    let width: usize = if codes.iter().all(|c| *c < 65536) { 2 } else { 4 };
    let (nc, nf) = (codes.len(), floats.len());
    let mut payload = vec![0u8; 1 + nc * width + nf * 4];
    payload[0] = width as u8;
    for (i, c) in codes.iter().enumerate() {
        for (b, byte) in c.to_le_bytes()[..width].iter().enumerate() {
            payload[1 + b * nc + i] = *byte;
        }
    }
    let base = 1 + nc * width;
    for (i, v) in floats.iter().enumerate() {
        for (b, byte) in v.to_le_bytes().iter().enumerate() {
            payload[base + b * nf + i] = *byte;
        }
    }
    let mut enc = ZlibEncoder::new(Vec::with_capacity(payload.len() / 3), Compression::new(5));
    enc.write_all(&payload).map_err(err)?;
    Ok((enc.finish().map_err(err)?, filled))
}

pub fn write_file(path: &Path, g: &WrfGrid, domain: &str, member: &str, valid: i64, tile: usize) -> Res<u64> {
    if g.nx < 2 || g.ny < 2 || g.nz < 2 {
        return Err(format!("{domain}: a {}x{}x{} grid has no columns to store", g.nx, g.ny, g.nz));
    }
    let tiles_x = (g.nx - 1).div_ceil(tile);
    let tiles_y = (g.ny - 1).div_ceil(tile);
    let td2: Vec<f64> = (0..g.nx * g.ny).into_par_iter().map(|i| g.td2(i)).collect();
    let encoded: Vec<(Vec<u8>, usize)> = (0..tiles_x * tiles_y)
        .into_par_iter()
        .map(|t| {
            let (x0, x1) = span(t % tiles_x, tile, g.nx);
            let (y0, y1) = span(t / tiles_x, tile, g.ny);
            encode_tile(g, &td2, x0, x1, y0, y1)
        })
        .collect::<Res<_>>()?;
    let filled: usize = encoded.iter().map(|(_, f)| f).sum();
    let blobs: Vec<Vec<u8>> = encoded.into_iter().map(|(b, _)| b).collect();
    // The locator: lat/lon every `tile` grid points (and the last row and column), so a reader finds the tile for
    // a latitude and longitude from the header alone.
    let pick = |n: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).step_by(tile).collect();
        if *v.last().unwrap() != n - 1 {
            v.push(n - 1);
        }
        v
    };
    let (ix, iy) = (pick(g.nx), pick(g.ny));
    let round = |x: f64| (x * 1e5).round() / 1e5;
    let mut llat = Vec::with_capacity(ix.len() * iy.len());
    let mut llon = Vec::with_capacity(ix.len() * iy.len());
    for &y in &iy {
        for &x in &ix {
            llat.push(round(g.lat[y * g.nx + x]));
            llon.push(round(g.lon[y * g.nx + x]));
        }
    }
    let mut header = json!({
        "schema": "gpuwm-columns.v1",
        "domain": domain,
        "member": member,
        "valid": chrono::DateTime::from_timestamp(valid, 0).unwrap().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        "nx": g.nx, "ny": g.ny, "nz": g.nz,
        "tile": tile, "tiles_x": tiles_x, "tiles_y": tiles_y,
        "fields3d": FIELDS_3D, "fields2d": FIELDS_2D,
        "codec3d": "quantized, differenced up the column, zigzag, u16 or u32 (first payload byte)",
        "quanta3d": QUANTA_3D, "dtype2d": "f32", "codec": "shuffle-zlib", "filled": filled,
        "index_offset": 0,
        "locator": {"step": tile, "ix": ix, "iy": iy, "lat": llat, "lon": llon},
    });
    // The index offset depends on the header's own length: leave room for the number, then pad to it.
    let probe = serde_json::to_vec(&header).map_err(err)?;
    let index_offset = 12 + probe.len() + 24;
    header["index_offset"] = json!(index_offset);
    let mut head = serde_json::to_vec(&header).map_err(err)?;
    head.resize(index_offset - 12, b' ');
    let total = index_offset + blobs.len() * 12 + blobs.iter().map(|b| b.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(head.len() as u32).to_le_bytes());
    out.extend_from_slice(&head);
    let mut offset = (index_offset + blobs.len() * 12) as u64;
    for b in &blobs {
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        offset += b.len() as u64;
    }
    for b in &blobs {
        out.extend_from_slice(b);
    }
    // Written beside and renamed in, so an uploader never ships half a file.
    let tmp = path.with_extension("wcol.tmp");
    std::fs::write(&tmp, &out).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(err)?;
    Ok(out.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, nz: usize) -> WrfGrid {
        let n = nx * ny;
        let v3 = |f: &dyn Fn(usize, usize, usize) -> f64| -> std::sync::Arc<[f64]> {
            (0..nz * n).map(|i| f(i / n, (i % n) % nx, (i % n) / nx)).collect::<Vec<_>>().into()
        };
        let v2 = |f: &dyn Fn(usize, usize) -> f64| -> std::sync::Arc<[f64]> {
            (0..n).map(|i| f(i % nx, i / nx)).collect::<Vec<_>>().into()
        };
        WrfGrid {
            nx,
            ny,
            nz,
            lat: v2(&|_, y| 35.0 + y as f64 * 0.03),
            lon: v2(&|x, _| -98.0 + x as f64 * 0.03),
            p: v3(&|k, _, _| 100_000.0 - k as f64 * 8_000.0),
            z: v3(&|k, _, _| 300.0 + k as f64 * 800.0),
            t: v3(&|k, x, _| 25.0 - k as f64 * 6.5 + x as f64 * 0.01),
            td: v3(&|k, _, y| 18.0 - k as f64 * 8.0 + y as f64 * 0.01),
            u: v3(&|k, _, _| 2.0 + k as f64 * 3.0),
            v: v3(&|k, _, _| 5.0 - k as f64),
            omega: v3(&|_, _, _| f64::NAN),
            ps: v2(&|_, _| 98_000.0),
            terrain: v2(&|_, _| 250.0),
            t2: v2(&|_, _| 300.0),
            q2: v2(&|_, _| 0.012),
            u10: v2(&|_, _| 1.0),
            v10: v2(&|_, _| 2.0),
        }
    }

    /// The layout a reader relies on: magic, header, an index whose tiles run back to back to the end of the file,
    /// one more tile than columns per tile row when the grid does not divide, and a missing omega counted as filled.
    #[test]
    fn store_layout_is_what_readers_read() {
        let dir = std::env::temp_dir().join(format!("rw-columns-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wcol");
        let g = grid(37, 20, 6);
        let bytes = write_file(&path, &g, "d01", "det", 1_720_972_800, 16).unwrap();
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw.len() as u64, bytes);
        assert_eq!(&raw[..8], MAGIC);
        let h = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let meta: serde_json::Value = serde_json::from_slice(raw[12..12 + h].trim_ascii_end()).unwrap();
        assert_eq!(meta["tiles_x"], 3);
        assert_eq!(meta["tiles_y"], 2);
        // Counted per stored value, and tiles overlap by one column: (17 + 17 + 5) x (17 + 4) columns x 6 levels.
        assert_eq!(meta["filled"], 39 * 21 * 6);
        assert_eq!(meta["valid"], "2024-07-14T16:00:00Z");
        let index = meta["index_offset"].as_u64().unwrap() as usize;
        assert_eq!(index, 12 + h);
        let mut end = index + 6 * 12;
        for t in 0..6 {
            let e = &raw[index + t * 12..index + t * 12 + 12];
            let off = u64::from_le_bytes(e[..8].try_into().unwrap()) as usize;
            let len = u32::from_le_bytes(e[8..].try_into().unwrap()) as usize;
            assert_eq!(off, end);
            end += len;
        }
        assert_eq!(end, raw.len());
        std::fs::remove_dir_all(&dir).ok();
    }
}
