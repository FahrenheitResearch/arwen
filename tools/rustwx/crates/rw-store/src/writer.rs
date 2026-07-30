//! Hour-file writer: assembles 2D surface variables into one rw-store file.
//!
//! File layout: `[64B header][meta JSON][index records, 64B each, sorted by
//! sort_key()][payload]`. Record offsets are absolute file offsets. Tiles
//! are encoded in parallel with rayon but staged and sorted deterministically,
//! so identical inputs always produce byte-identical files.
//!
//! Memory shape: with [`HourWriter::with_spill_dir`] every encoded chunk's
//! compressed bytes are appended to a temp spill file as they are staged
//! (only the 64-byte records stay resident), and [`HourWriter::finish`]
//! streams header + meta + index + payload straight into the atomic temp
//! file instead of assembling the whole hour (~650 MB at HRRR size) in one
//! `Vec`. The output bytes are identical either way: chunk payload bytes
//! never depend on staging location, and the index/payload order is the
//! same deterministic `sort_key()` order.
//!
//! Variable ids are assigned in add order. `add_pressure3d_deferred` lets a
//! caller encode a variable's chunks early (freeing the raw planes) while
//! its id — and therefore its position in the meta/index — is assigned at
//! `finish()` AFTER every normally-added variable, exactly as if it had
//! been added last. The store ingest uses this to encode the extracted
//! volumes before the derived/heavy compute stages run, without changing
//! one byte of the resulting file.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::atomic::atomic_write_with;
use crate::codec::{encode_affine_i16, encode_f32_tile};
use crate::error::{RwResult, RwStoreError};
use crate::format::{
    CODEC_2D, CODEC_3D, COL_X, COL_Y, KIND_COLUMN3D, KIND_TILE2D, RwsChunking, RwsExactTime,
    RwsHourMeta, RwsVariableMeta, RwsWriterInfo, SCHEMA_HOUR, SCHEMA_HOUR_V2, TILE_X, TILE_Y,
};
use crate::header::RwsHeader;
use crate::index::ChunkRecord;

/// zstd compression level for dense tile payloads (matches CODEC_2D name).
const ZSTD_LEVEL: i32 = 1;

/// One encoded chunk staged for assembly; `record.offset` is assigned in
/// [`HourWriter::finish`] once the global chunk order is known. The
/// compressed bytes live either inline (`compressed`) or in the spill file
/// at `spill_offset` (in which case `compressed` is empty and the byte
/// count is `record.len`).
struct StagedChunk {
    record: ChunkRecord,
    compressed: Vec<u8>,
    spill_offset: u64,
    /// `Some(k)` for chunks of the k-th deferred variable: `record.var_id`
    /// is rewritten to the final id in `finish()`.
    deferred_index: Option<u16>,
}

/// The spill file holding staged compressed chunk bytes. Removed on drop,
/// so an abandoned writer (error or cancel) never leaks the temp file.
struct SpillFile {
    path: PathBuf,
    file: fs::File,
    cursor: u64,
}

impl SpillFile {
    fn create(dir: &Path) -> RwResult<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        fs::create_dir_all(dir)?;
        let path = dir.join(format!(
            ".rws-spill-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let file = fs::OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)?;
        Ok(Self {
            path,
            file,
            cursor: 0,
        })
    }

    /// Append one chunk's compressed bytes; returns its spill offset.
    fn append(&mut self, bytes: &[u8]) -> RwResult<u64> {
        let offset = self.cursor;
        self.file.write_all(bytes)?;
        self.cursor += bytes.len() as u64;
        Ok(offset)
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Incremental builder for a single per-hour store file.
pub struct HourWriter {
    model: String,
    run: String,
    forecast_hour: u16,
    exact_time: Option<RwsExactTime>,
    nx: usize,
    ny: usize,
    grid_hash: String,
    writer_build: String,
    variables: Vec<RwsVariableMeta>,
    /// Deferred variables (ids assigned in `finish()` after `variables`),
    /// with their meta carrying a placeholder id until then.
    deferred_variables: Vec<RwsVariableMeta>,
    chunks: Vec<StagedChunk>,
    spill_dir: Option<PathBuf>,
    spill: Option<SpillFile>,
}

impl HourWriter {
    pub fn new(
        model: &str,
        run: &str,
        forecast_hour: u16,
        nx: usize,
        ny: usize,
        grid_hash: &str,
        writer_build: &str,
    ) -> Self {
        Self::new_with_time(
            model,
            run,
            forecast_hour,
            None,
            nx,
            ny,
            grid_hash,
            writer_build,
        )
    }

    /// Build a v2 hour whose numeric key is an ordinal storage slot and whose
    /// physical lead/valid time is exact to the second.
    pub fn new_exact(
        model: &str,
        run: &str,
        storage_slot: u16,
        exact_time: RwsExactTime,
        nx: usize,
        ny: usize,
        grid_hash: &str,
        writer_build: &str,
    ) -> Self {
        Self::new_with_time(
            model,
            run,
            storage_slot,
            Some(exact_time),
            nx,
            ny,
            grid_hash,
            writer_build,
        )
    }

    fn new_with_time(
        model: &str,
        run: &str,
        forecast_hour: u16,
        exact_time: Option<RwsExactTime>,
        nx: usize,
        ny: usize,
        grid_hash: &str,
        writer_build: &str,
    ) -> Self {
        Self {
            model: model.to_string(),
            run: run.to_string(),
            forecast_hour,
            exact_time,
            nx,
            ny,
            grid_hash: grid_hash.to_string(),
            writer_build: writer_build.to_string(),
            variables: Vec::new(),
            deferred_variables: Vec::new(),
            chunks: Vec::new(),
            spill_dir: None,
            spill: None,
        }
    }

    /// Spill staged compressed chunk bytes to a temp file under `dir`
    /// (created lazily on the first add) instead of holding them in memory
    /// until `finish()`. Output bytes are unaffected.
    pub fn with_spill_dir(mut self, dir: &Path) -> Self {
        self.spill_dir = Some(dir.to_path_buf());
        self
    }

    /// Move freshly encoded chunks into the staged set, spilling their
    /// payload bytes when a spill dir is configured.
    fn stage_chunks(&mut self, encoded: Vec<StagedChunk>) -> RwResult<()> {
        if let Some(dir) = self.spill_dir.as_deref() {
            if self.spill.is_none() {
                self.spill = Some(SpillFile::create(dir)?);
            }
        }
        match self.spill.as_mut() {
            Some(spill) => {
                for mut chunk in encoded {
                    debug_assert_eq!(chunk.compressed.len() as u32, chunk.record.len);
                    chunk.spill_offset = spill.append(&chunk.compressed)?;
                    chunk.compressed = Vec::new();
                    self.chunks.push(chunk);
                }
            }
            None => self.chunks.extend(encoded),
        }
        Ok(())
    }

    /// Add a 2D surface field (row-major, `ny * nx` values). Tiles are
    /// encoded in parallel; returns the assigned variable id.
    pub fn add_surface2d(
        &mut self,
        name: &str,
        units: &str,
        selector: serde_json::Value,
        values: &[f32],
    ) -> RwResult<u16> {
        let expected = self.nx * self.ny;
        if values.len() != expected {
            return Err(RwStoreError::Format(format!(
                "variable '{name}': expected {expected} values ({} x {}), got {}",
                self.ny,
                self.nx,
                values.len()
            )));
        }
        let var_id = self.next_var_id(name)?;

        let tiles_y = self.ny.div_ceil(TILE_Y);
        let tiles_x = self.nx.div_ceil(TILE_X);
        let tile_coords: Vec<(usize, usize)> = (0..tiles_y)
            .flat_map(|ty| (0..tiles_x).map(move |tx| (ty, tx)))
            .collect();

        let nx = self.nx;
        let ny = self.ny;
        // Parallel encode; collect preserves tile_coords order so staging
        // (and therefore the final file) is independent of rayon scheduling.
        let encoded: Vec<StagedChunk> = tile_coords
            .par_iter()
            .map(|&(ty, tx)| -> RwResult<StagedChunk> {
                let y0 = ty * TILE_Y;
                let x0 = tx * TILE_X;
                let y1 = (y0 + TILE_Y).min(ny);
                let x1 = (x0 + TILE_X).min(nx);
                let mut tile_values = Vec::with_capacity((y1 - y0) * (x1 - x0));
                for y in y0..y1 {
                    tile_values.extend_from_slice(&values[y * nx + x0..y * nx + x1]);
                }
                let chunk = encode_f32_tile(&tile_values);
                let compressed = if chunk.payload.is_empty() {
                    Vec::new()
                } else {
                    zstd::stream::encode_all(&chunk.payload[..], ZSTD_LEVEL)?
                };
                Ok(StagedChunk {
                    record: ChunkRecord {
                        var_id,
                        kind: KIND_TILE2D,
                        flags: chunk.flags,
                        tile_y: ty as u32,
                        tile_x: tx as u32,
                        offset: 0, // assigned in finish()
                        len: compressed.len() as u32,
                        raw_len: chunk.payload.len() as u32,
                        center: chunk.center,
                        scale: chunk.scale,
                        min: chunk.min,
                        max: chunk.max,
                        valid_count: chunk.valid_count,
                    },
                    compressed,
                    spill_offset: 0,
                    deferred_index: None,
                })
            })
            .collect::<RwResult<Vec<StagedChunk>>>()?;

        self.stage_chunks(encoded)?;
        self.variables.push(RwsVariableMeta {
            id: var_id,
            name: name.to_string(),
            units: units.to_string(),
            kind: "surface2d".to_string(),
            codec: CODEC_2D.to_string(),
            levels_hpa: Vec::new(),
            selector,
        });
        Ok(var_id)
    }

    /// Add a 3D pressure-level field as `[y][x][z]` column chunks: within
    /// each 16x16 chunk the L level values of a column are contiguous, so a
    /// sounding pull decodes one chunk and slices one run. `levels_hpa` must
    /// be strictly descending (1000 first) and `level_planes` holds one
    /// row-major `ny * nx` plane per level, in that same order. The whole
    /// chunk is affine-i16 quantized (one center/scale) then zstd-1.
    pub fn add_pressure3d(
        &mut self,
        name: &str,
        units: &str,
        selector: serde_json::Value,
        levels_hpa: &[u16],
        level_planes: &[&[f32]],
    ) -> RwResult<u16> {
        self.validate_pressure3d(name, levels_hpa, level_planes)?;
        let var_id = self.next_var_id(name)?;
        self.encode_pressure3d_chunks(var_id, None, level_planes)?;
        self.variables.push(RwsVariableMeta {
            id: var_id,
            name: name.to_string(),
            units: units.to_string(),
            kind: "pressure3d".to_string(),
            codec: CODEC_3D.to_string(),
            levels_hpa: levels_hpa.to_vec(),
            selector,
        });
        Ok(var_id)
    }

    /// [`Self::add_pressure3d`] with id assignment deferred to `finish()`:
    /// chunks are encoded (and spilled) NOW, but the variable is numbered
    /// after every normally-added variable, in deferred-add order — the
    /// file comes out byte-identical to adding it last. This lets the
    /// store ingest free its raw volume planes before the compute stages
    /// run while keeping the historical variable order
    /// (fields, derived, heavy, volumes).
    pub fn add_pressure3d_deferred(
        &mut self,
        name: &str,
        units: &str,
        selector: serde_json::Value,
        levels_hpa: &[u16],
        level_planes: &[&[f32]],
    ) -> RwResult<()> {
        self.validate_pressure3d(name, levels_hpa, level_planes)?;
        // Name-uniqueness must span both sets; the id itself is a
        // placeholder rewritten in finish().
        self.next_var_id(name)?;
        let deferred_index = u16::try_from(self.deferred_variables.len()).map_err(|_| {
            RwStoreError::Format(format!(
                "too many deferred variables: index for '{name}' exceeds u16"
            ))
        })?;
        self.encode_pressure3d_chunks(deferred_index, Some(deferred_index), level_planes)?;
        self.deferred_variables.push(RwsVariableMeta {
            id: deferred_index, // placeholder; final id assigned in finish()
            name: name.to_string(),
            units: units.to_string(),
            kind: "pressure3d".to_string(),
            codec: CODEC_3D.to_string(),
            levels_hpa: levels_hpa.to_vec(),
            selector,
        });
        Ok(())
    }

    fn validate_pressure3d(
        &self,
        name: &str,
        levels_hpa: &[u16],
        level_planes: &[&[f32]],
    ) -> RwResult<()> {
        if levels_hpa.is_empty() {
            return Err(RwStoreError::Format(format!(
                "variable '{name}': levels_hpa must not be empty"
            )));
        }
        if let Some(pair) = levels_hpa.windows(2).find(|pair| pair[0] <= pair[1]) {
            return Err(RwStoreError::Format(format!(
                "variable '{name}': levels_hpa must be strictly descending, found {} then {}",
                pair[0], pair[1]
            )));
        }
        if level_planes.len() != levels_hpa.len() {
            return Err(RwStoreError::Format(format!(
                "variable '{name}': {} level planes for {} levels",
                level_planes.len(),
                levels_hpa.len()
            )));
        }
        let expected = self.nx * self.ny;
        for (k, plane) in level_planes.iter().enumerate() {
            if plane.len() != expected {
                return Err(RwStoreError::Format(format!(
                    "variable '{name}' level {} ({} hPa): expected {expected} values \
                     ({} x {}), got {}",
                    k,
                    levels_hpa[k],
                    self.ny,
                    self.nx,
                    plane.len()
                )));
            }
        }
        Ok(())
    }

    /// Encode one 3D variable's column chunks and stage them under
    /// `var_id` (the final id, or the placeholder for deferred variables).
    fn encode_pressure3d_chunks(
        &mut self,
        var_id: u16,
        deferred_index: Option<u16>,
        level_planes: &[&[f32]],
    ) -> RwResult<()> {
        let chunks_y = self.ny.div_ceil(COL_Y);
        let chunks_x = self.nx.div_ceil(COL_X);
        let chunk_coords: Vec<(usize, usize)> = (0..chunks_y)
            .flat_map(|cy| (0..chunks_x).map(move |cx| (cy, cx)))
            .collect();

        let nx = self.nx;
        let ny = self.ny;
        let levels = level_planes.len();
        // Parallel encode; collect preserves chunk_coords order so staging
        // (and therefore the final file) is independent of rayon scheduling.
        let encoded: Vec<StagedChunk> = chunk_coords
            .par_iter()
            .map(|&(cy, cx)| -> RwResult<StagedChunk> {
                let y0 = cy * COL_Y;
                let x0 = cx * COL_X;
                let y1 = (y0 + COL_Y).min(ny);
                let x1 = (x0 + COL_X).min(nx);
                // Gather [y][x][z]: for each footprint cell, its L level
                // values are contiguous, ordered by levels_hpa order.
                let mut buffer = Vec::with_capacity((y1 - y0) * (x1 - x0) * levels);
                for gy in y0..y1 {
                    for gx in x0..x1 {
                        let cell = gy * nx + gx;
                        for plane in level_planes {
                            buffer.push(plane[cell]);
                        }
                    }
                }
                let chunk = encode_affine_i16(&buffer)?;
                let compressed = if chunk.payload.is_empty() {
                    Vec::new()
                } else {
                    zstd::stream::encode_all(&chunk.payload[..], ZSTD_LEVEL)?
                };
                Ok(StagedChunk {
                    record: ChunkRecord {
                        var_id,
                        kind: KIND_COLUMN3D,
                        flags: chunk.flags,
                        tile_y: cy as u32,
                        tile_x: cx as u32,
                        offset: 0, // assigned in finish()
                        len: compressed.len() as u32,
                        raw_len: chunk.payload.len() as u32,
                        center: chunk.center,
                        scale: chunk.scale,
                        min: chunk.min,
                        max: chunk.max,
                        valid_count: chunk.valid_count,
                    },
                    compressed,
                    spill_offset: 0,
                    deferred_index,
                })
            })
            .collect::<RwResult<Vec<StagedChunk>>>()?;

        self.stage_chunks(encoded)
    }

    /// Reserve the next variable id, enforcing name uniqueness across all
    /// kinds (normal and deferred) and the u16 id budget.
    fn next_var_id(&self, name: &str) -> RwResult<u16> {
        if self
            .variables
            .iter()
            .chain(self.deferred_variables.iter())
            .any(|var| var.name == name)
        {
            return Err(RwStoreError::Format(format!(
                "duplicate variable name '{name}'"
            )));
        }
        u16::try_from(self.variables.len()).map_err(|_| {
            RwStoreError::Format(format!(
                "too many variables: var id for '{name}' exceeds u16"
            ))
        })
    }

    /// Assemble and atomically write the hour file, returning its metadata.
    /// Deferred variables receive their final ids here (after every
    /// normally-added variable, in deferred-add order); the index/payload
    /// are then emitted in the same deterministic `sort_key()` order as
    /// always, streamed from memory or the spill file straight into the
    /// atomic temp file.
    pub fn finish(mut self, path: &Path) -> RwResult<RwsHourMeta> {
        if self.nx == 0 || self.ny == 0 {
            return Err(RwStoreError::Format(format!(
                "degenerate grid {}x{} (nx and ny must be nonzero)",
                self.nx, self.ny
            )));
        }
        if self
            .exact_time
            .is_some_and(|time| time.origin_unix().is_none())
        {
            return Err(RwStoreError::Meta(
                "exact hour cannot represent valid_unix - lead_seconds".to_string(),
            ));
        }

        // Final ids for deferred variables: numbered after the normal set.
        let normal_count = self.variables.len();
        let mut variables = self.variables;
        for (index, mut var) in self.deferred_variables.into_iter().enumerate() {
            let id = normal_count + index;
            var.id = u16::try_from(id).map_err(|_| {
                RwStoreError::Format(format!(
                    "too many variables: var id for '{}' exceeds u16",
                    var.name
                ))
            })?;
            variables.push(var);
        }
        for chunk in &mut self.chunks {
            if let Some(index) = chunk.deferred_index {
                chunk.record.var_id = u16::try_from(normal_count + index as usize)
                    .expect("deferred var id bounds checked above");
            }
        }
        self.chunks.sort_by_key(|chunk| chunk.record.sort_key());

        let exact_time = self.exact_time;
        let meta = RwsHourMeta {
            schema: if exact_time.is_some() {
                SCHEMA_HOUR_V2.to_string()
            } else {
                SCHEMA_HOUR.to_string()
            },
            model: self.model,
            run: self.run,
            forecast_hour: self.forecast_hour,
            lead_seconds: exact_time.map(|time| time.lead_seconds),
            valid_unix: exact_time.map(|time| time.valid_unix),
            nx: self.nx,
            ny: self.ny,
            grid_hash: self.grid_hash,
            variables,
            chunking: RwsChunking {
                tile_y: TILE_Y,
                tile_x: TILE_X,
                col_y: COL_Y,
                col_x: COL_X,
            },
            writer: RwsWriterInfo {
                name: "rw-store".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                build: self.writer_build,
            },
        };
        let meta_bytes =
            serde_json::to_vec(&meta).map_err(|err| RwStoreError::Meta(err.to_string()))?;
        let meta_len = u32::try_from(meta_bytes.len()).map_err(|_| {
            RwStoreError::Format(format!("meta JSON too large: {} bytes", meta_bytes.len()))
        })?;
        let header = RwsHeader::for_layout(meta_len, self.chunks.len() as u64);

        // Assign absolute payload offsets cursor-style in sorted order.
        // EMPTY/CONSTANT chunks have len 0; their offset is wherever the
        // cursor currently sits (value unused by readers).
        let mut cursor = header.payload_offset;
        for chunk in &mut self.chunks {
            chunk.record.offset = cursor;
            cursor += u64::from(chunk.record.len);
        }
        let total_len = cursor;

        // Stream header + meta + index + payload into the atomic temp file
        // (sorted order, exactly the bytes the historical Vec assembly
        // produced). Spilled payloads are read back per chunk; the spill
        // file was written in add order, so sorted emission seeks — the
        // pages are warm from the just-finished writes.
        let mut spill = self.spill;
        let chunks = self.chunks;
        let result = atomic_write_with(path, |writer| {
            let mut written: u64 = 0;
            writer.write_all(&header.pack())?;
            written += 64;
            writer.write_all(&meta_bytes)?;
            written += meta_bytes.len() as u64;
            let mut record_buf = Vec::with_capacity(64 * chunks.len());
            for chunk in &chunks {
                chunk.record.pack_into(&mut record_buf);
            }
            writer.write_all(&record_buf)?;
            written += record_buf.len() as u64;
            debug_assert_eq!(written, header.payload_offset);
            let mut payload_buf: Vec<u8> = Vec::new();
            for chunk in &chunks {
                let len = chunk.record.len as usize;
                if len == 0 {
                    continue;
                }
                debug_assert_eq!(chunk.record.offset, written);
                match spill.as_mut() {
                    Some(spill) => {
                        payload_buf.resize(len, 0);
                        spill.file.seek(SeekFrom::Start(chunk.spill_offset))?;
                        spill.file.read_exact(&mut payload_buf)?;
                        writer.write_all(&payload_buf)?;
                    }
                    None => writer.write_all(&chunk.compressed)?,
                }
                written += len as u64;
            }
            debug_assert_eq!(written, total_len);
            Ok(())
        });
        // Explicit for clarity: dropping the spill handle removes the temp
        // spill file whether the write succeeded or failed.
        drop(spill);
        result?;
        Ok(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RwStoreError;
    use crate::format::{
        CODEC_2D, FLAG_CONSTANT, FLAG_EMPTY, KIND_TILE2D, RwsExactTime, RwsHourMeta, SCHEMA_HOUR,
        SCHEMA_HOUR_V2, TILE_X, TILE_Y,
    };
    use crate::header::RwsHeader;
    use crate::index::ChunkRecord;
    use crate::reader::HourReader;
    use std::fs;
    use std::path::{Path, PathBuf};

    const NX: usize = 600; // columns -> x tiles of 256, 256, 88
    const NY: usize = 500; // rows    -> y tiles of 256, 244
    const TILES_PER_VAR: usize = 6; // 3 x-tiles * 2 y-tiles

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-store-writer-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Var A: smooth field, with tile (0,0) all-NaN (EMPTY) and tile (0,1)
    /// all 42.0 (CONSTANT). Both regions are full 256x256 aligned tiles.
    fn grid_a() -> Vec<f32> {
        let mut values: Vec<f32> = (0..NY)
            .flat_map(|y| (0..NX).map(move |x| 0.01 * x as f32 + 0.02 * y as f32))
            .collect();
        for y in 0..TILE_Y {
            for x in 0..TILE_X {
                values[y * NX + x] = f32::NAN;
            }
            for x in TILE_X..2 * TILE_X {
                values[y * NX + x] = 42.0;
            }
        }
        values
    }

    /// Var B: varying everywhere; every tile must encode dense.
    fn grid_b() -> Vec<f32> {
        (0..NY)
            .flat_map(|y| (0..NX).map(move |x| 100.0 + 0.5 * x as f32 - 0.25 * y as f32))
            .collect()
    }

    fn tile_dims(tile_y: u32, tile_x: u32) -> (usize, usize) {
        let y0 = tile_y as usize * TILE_Y;
        let x0 = tile_x as usize * TILE_X;
        ((NY - y0).min(TILE_Y), (NX - x0).min(TILE_X))
    }

    fn write_sample(path: &Path) -> RwsHourMeta {
        let mut writer = HourWriter::new(
            "hrrr",
            "20260609_12z",
            6,
            NX,
            NY,
            "gridhash-test",
            "test-build",
        );
        let id_a = writer
            .add_surface2d(
                "temp_2m",
                "K",
                serde_json::json!({"grib_short_name": "TMP", "level": "2 m above ground"}),
                &grid_a(),
            )
            .unwrap();
        let id_b = writer
            .add_surface2d(
                "dewpoint_2m",
                "K",
                serde_json::json!({"grib_short_name": "DPT", "level": "2 m above ground"}),
                &grid_b(),
            )
            .unwrap();
        assert_eq!((id_a, id_b), (0, 1), "var ids assigned sequentially");
        writer.finish(path).unwrap()
    }

    #[test]
    fn writes_two_var_hour_file_with_correct_raw_layout() {
        let dir = test_dir("layout");
        let path = dir.join("hour.rws");
        let returned_meta = write_sample(&path);

        let bytes = fs::read(&path).unwrap();
        let header = RwsHeader::parse(&bytes).unwrap();

        // Meta JSON.
        let meta_end = 64 + header.meta_len as usize;
        let meta: RwsHourMeta = serde_json::from_slice(&bytes[64..meta_end]).unwrap();
        assert_eq!(meta, returned_meta, "finish() must return the written meta");
        assert_eq!(meta.schema, SCHEMA_HOUR);
        assert_eq!(meta.model, "hrrr");
        assert_eq!(meta.run, "20260609_12z");
        assert_eq!(meta.forecast_hour, 6);
        assert_eq!(meta.nx, NX);
        assert_eq!(meta.ny, NY);
        assert_eq!(meta.grid_hash, "gridhash-test");
        assert_eq!(meta.chunking.tile_y, TILE_Y);
        assert_eq!(meta.chunking.tile_x, TILE_X);
        assert_eq!(meta.writer.name, "rw-store");
        assert_eq!(meta.writer.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(meta.writer.build, "test-build");

        assert_eq!(meta.variables.len(), 2);
        assert_ne!(meta.variables[0].id, meta.variables[1].id);
        for var in &meta.variables {
            assert_eq!(var.kind, "surface2d");
            assert_eq!(var.codec, CODEC_2D);
            assert!(var.levels_hpa.is_empty());
        }
        assert_eq!(meta.variables[0].name, "temp_2m");
        assert_eq!(meta.variables[1].name, "dewpoint_2m");

        // Index records.
        assert_eq!(header.index_count as usize, 2 * TILES_PER_VAR, "12 chunks");
        assert_eq!(
            header.index_offset as usize, meta_end,
            "index follows meta JSON"
        );
        let records: Vec<ChunkRecord> = (0..header.index_count as usize)
            .map(|i| {
                let start = header.index_offset as usize + i * 64;
                ChunkRecord::unpack(&bytes[start..start + 64]).unwrap()
            })
            .collect();
        for pair in records.windows(2) {
            assert!(
                pair[0].sort_key() < pair[1].sort_key(),
                "records must be strictly sorted by sort_key"
            );
        }
        assert!(records.iter().all(|r| r.kind == KIND_TILE2D));

        // payload_offset matches the fixed layout.
        assert_eq!(
            header.payload_offset,
            header.index_offset + header.index_count * 64
        );

        // NaN tile (var 0, tile 0,0) -> EMPTY.
        let empty = records
            .iter()
            .find(|r| r.var_id == 0 && r.tile_y == 0 && r.tile_x == 0)
            .expect("record for var 0 tile (0,0)");
        assert_ne!(empty.flags & FLAG_EMPTY, 0, "NaN tile must be FLAG_EMPTY");
        assert_eq!(empty.len, 0);
        assert_eq!(empty.raw_len, 0);

        // Constant tile (var 0, tile 0,1) -> CONSTANT with center 42.0.
        let constant = records
            .iter()
            .find(|r| r.var_id == 0 && r.tile_y == 0 && r.tile_x == 1)
            .expect("record for var 0 tile (0,1)");
        assert_ne!(
            constant.flags & FLAG_CONSTANT,
            0,
            "42.0 tile must be CONSTANT"
        );
        assert_eq!(constant.len, 0);
        assert_eq!(constant.center, 42.0);

        // Dense records: 10 of 12; compressed payloads in bounds with exact raw sizes.
        let dense: Vec<&ChunkRecord> = records
            .iter()
            .filter(|r| r.flags & (FLAG_EMPTY | FLAG_CONSTANT) == 0)
            .collect();
        assert_eq!(dense.len(), 10, "4 dense tiles for var A + 6 for var B");
        for record in &dense {
            let (rows, cols) = tile_dims(record.tile_y, record.tile_x);
            assert!(record.len > 0, "dense chunk must have compressed payload");
            assert_eq!(
                record.raw_len as usize,
                rows * cols * 4,
                "raw_len must equal tile f32 byte count for tile ({},{})",
                record.tile_y,
                record.tile_x
            );
            assert!(record.offset >= header.payload_offset);
            assert!(record.offset + record.len as u64 <= bytes.len() as u64);
        }

        // Spot-check one dense payload: var B edge tile (1,2) = 244x88,
        // decompresses to the expected raw f32 bytes for that window.
        let spot = records
            .iter()
            .find(|r| r.var_id == 1 && r.tile_y == 1 && r.tile_x == 2)
            .expect("record for var 1 tile (1,2)");
        let compressed = &bytes[spot.offset as usize..spot.offset as usize + spot.len as usize];
        let raw = zstd::stream::decode_all(compressed).unwrap();
        assert_eq!(raw.len(), spot.raw_len as usize);
        let grid = grid_b();
        let mut expected = Vec::with_capacity(244 * 88 * 4);
        for y in 256..NY {
            for x in 512..NX {
                expected.extend_from_slice(&grid[y * NX + x].to_le_bytes());
            }
        }
        assert_eq!(raw, expected, "tile payload must be row-major within tile");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn exact_writer_emits_v2_schema_and_complete_time_pair() {
        let dir = test_dir("exact-meta");
        let path = dir.join("f007.rws");
        let exact = RwsExactTime {
            lead_seconds: 2_700,
            valid_unix: 1_700_002_700,
        };
        let meta = HourWriter::new_exact(
            "wrf",
            "research",
            7,
            exact,
            1,
            1,
            "gridhash-test",
            "test-build",
        )
        .finish(&path)
        .unwrap();
        assert_eq!(meta.schema, SCHEMA_HOUR_V2);
        assert_eq!(meta.forecast_hour, 7);
        assert_eq!(meta.exact_time(), Some(exact));
        assert_eq!(HourReader::open(&path).unwrap().meta(), &meta);

        let invalid = dir.join("invalid.rws");
        let error = HourWriter::new_exact(
            "wrf",
            "research",
            8,
            RwsExactTime {
                lead_seconds: u64::MAX,
                valid_unix: 0,
            },
            1,
            1,
            "gridhash-test",
            "test-build",
        )
        .finish(&invalid)
        .unwrap_err();
        assert!(error.to_string().contains("cannot represent"), "{error}");
        assert!(!invalid.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn same_input_produces_byte_identical_files() {
        let dir = test_dir("determinism");
        let path_one = dir.join("one.rws");
        let path_two = dir.join("two.rws");
        write_sample(&path_one);
        write_sample(&path_two);
        let bytes_one = fs::read(&path_one).unwrap();
        let bytes_two = fs::read(&path_two).unwrap();
        assert_eq!(
            bytes_one, bytes_two,
            "same inputs must produce byte-identical files"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The memory-diet seams must not change one output byte: a writer
    /// with spilled chunks and an early-added (deferred-id) volume produces
    /// exactly the bytes of the historical in-memory writer that adds the
    /// volume last.
    #[test]
    fn deferred_and_spilled_writes_match_the_one_shot_bytes() {
        let dir = test_dir("deferred-spill");
        let plane_high: Vec<f32> = grid_b();
        let plane_low: Vec<f32> = grid_b().iter().map(|v| v * 0.5 - 3.0).collect();

        // Historical shape: in-memory staging, volume added last.
        let path_one = dir.join("one.rws");
        let mut writer = HourWriter::new("hrrr", "run", 6, NX, NY, "hash", "build");
        writer
            .add_surface2d("temp_2m", "K", serde_json::json!({"a": 1}), &grid_a())
            .unwrap();
        writer
            .add_surface2d("dewpoint_2m", "K", serde_json::json!({"b": 2}), &grid_b())
            .unwrap();
        writer
            .add_pressure3d(
                "temperature_iso",
                "K",
                serde_json::json!({"vol": true}),
                &[1000, 500],
                &[&plane_low, &plane_high],
            )
            .unwrap();
        let meta_one = writer.finish(&path_one).unwrap();

        // Diet shape: spill dir + volume encoded FIRST with a deferred id.
        let path_two = dir.join("two.rws");
        let mut writer =
            HourWriter::new("hrrr", "run", 6, NX, NY, "hash", "build").with_spill_dir(&dir);
        writer
            .add_pressure3d_deferred(
                "temperature_iso",
                "K",
                serde_json::json!({"vol": true}),
                &[1000, 500],
                &[&plane_low, &plane_high],
            )
            .unwrap();
        writer
            .add_surface2d("temp_2m", "K", serde_json::json!({"a": 1}), &grid_a())
            .unwrap();
        writer
            .add_surface2d("dewpoint_2m", "K", serde_json::json!({"b": 2}), &grid_b())
            .unwrap();
        let meta_two = writer.finish(&path_two).unwrap();

        assert_eq!(meta_one, meta_two, "meta must be identical");
        let bytes_one = fs::read(&path_one).unwrap();
        let bytes_two = fs::read(&path_two).unwrap();
        assert_eq!(
            bytes_one, bytes_two,
            "deferred + spilled write must be byte-identical to the one-shot order"
        );
        let leftovers: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains("spill"))
            .collect();
        assert_eq!(
            leftovers,
            Vec::<String>::new(),
            "spill temp files must be cleaned up"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// An abandoned (dropped) spilling writer removes its spill temp file.
    #[test]
    fn dropped_spilling_writer_cleans_up_its_spill_file() {
        let dir = test_dir("spill-drop");
        let mut writer =
            HourWriter::new("hrrr", "run", 0, NX, NY, "hash", "build").with_spill_dir(&dir);
        writer
            .add_surface2d("temp_2m", "K", serde_json::Value::Null, &grid_b())
            .unwrap();
        drop(writer);
        let leftovers: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            leftovers,
            Vec::<String>::new(),
            "dropping the writer must remove the spill file"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_wrong_value_count() {
        let mut writer = HourWriter::new("hrrr", "run", 0, NX, NY, "hash", "build");
        let err = writer
            .add_surface2d("temp_2m", "K", serde_json::Value::Null, &[0.0; 17])
            .unwrap_err();
        assert!(
            matches!(err, RwStoreError::Format(_)),
            "expected Format error, got {err:?}"
        );
    }

    #[test]
    fn rejects_duplicate_variable_name() {
        let mut writer = HourWriter::new("hrrr", "run", 0, NX, NY, "hash", "build");
        writer
            .add_surface2d("temp_2m", "K", serde_json::Value::Null, &grid_b())
            .unwrap();
        let err = writer
            .add_surface2d("temp_2m", "K", serde_json::Value::Null, &grid_b())
            .unwrap_err();
        assert!(
            matches!(err, RwStoreError::Format(_)),
            "expected Format error, got {err:?}"
        );
    }
}
