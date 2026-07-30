//! Structural equivalence comparison for `.rws` hour files.
//!
//! Two hour files compare **equivalent** when their header `version` /
//! `index_count`, meta JSON (with `writer.build` masked out), index records
//! (with `offset` compared relative to each file's `payload_offset`), and
//! payload bytes all match.  This lets independent builds of the same inputs
//! be verified as deterministically equivalent even when the writer-build sha
//! differs (which shifts every absolute offset).
//!
//! The `assert-build` helpers ([`read_writer_build`] / [`build_matches`]) guard
//! against mislabeled baselines: verify the producer's build stamp BEFORE
//! trusting any comparison made with its output.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::format::{HEADER_LEN, INDEX_RECORD_LEN};
use crate::header::RwsHeader;
use crate::index::ChunkRecord;
use crate::run::MAX_RUN_MANIFEST_BYTES;

/// Hour metadata is JSON, not field payload. The writer's metadata is normally
/// measured in KiB; this ceiling leaves ample room for large variable catalogs
/// while preventing a corrupt header from making comparison/build inspection
/// allocate multiple GiB.
const MAX_HOUR_META_BYTES: u64 = 16 * 1024 * 1024;
/// Keep comparison within the same format envelope as `HourReader`.
const MAX_INDEX_RECORDS: u64 = 8_388_608;
/// Fixed-memory payload comparison buffer. Large legitimate hour files remain
/// supported: comparison streams through them rather than imposing a file cap.
const COMPARE_BUFFER_BYTES: usize = 64 * 1024;

/// A difference found between two files (or an I/O error while comparing).
#[derive(Debug)]
pub enum Difference {
    /// An I/O or format error prevented the comparison.
    Io(String),
    /// The files differ; the message describes the first difference found.
    Found(String),
}

struct OpenHour {
    file: File,
    file_len: u64,
    header: RwsHeader,
    path: PathBuf,
}

fn io_difference(path: &Path, action: &str, err: impl std::fmt::Display) -> Difference {
    Difference::Io(format!("{action} {}: {err}", path.display()))
}

/// Open once and retain the handle for the complete operation. Besides
/// avoiding path-replacement races, validating the header against the handle's
/// length lets all later reads use checked ranges.
fn open_hour(path: &Path) -> Result<OpenHour, Difference> {
    let mut file = File::open(path).map_err(|err| io_difference(path, "open", err))?;
    let file_len = file
        .metadata()
        .map_err(|err| io_difference(path, "metadata", err))?
        .len();
    let mut header_bytes = [0u8; HEADER_LEN];
    file.read_exact(&mut header_bytes)
        .map_err(|err| io_difference(path, "read header", err))?;
    let header = RwsHeader::parse(&header_bytes)
        .map_err(|err| Difference::Io(format!("header {}: {err}", path.display())))?;
    if header.index_count > MAX_INDEX_RECORDS {
        return Err(Difference::Io(format!(
            "{}: index_count {} exceeds supported limit {MAX_INDEX_RECORDS}",
            path.display(),
            header.index_count
        )));
    }
    if u64::from(header.meta_len) > MAX_HOUR_META_BYTES {
        return Err(Difference::Io(format!(
            "{}: hour metadata is {} bytes; limit is {MAX_HOUR_META_BYTES} bytes",
            path.display(),
            header.meta_len
        )));
    }
    if header.payload_offset > file_len {
        return Err(Difference::Io(format!(
            "{}: index ends at {} but file length is {file_len}",
            path.display(),
            header.payload_offset
        )));
    }
    Ok(OpenHour {
        file,
        file_len,
        header,
        path: path.to_path_buf(),
    })
}

fn read_region(
    file: &mut File,
    path: &Path,
    offset: u64,
    len: u64,
    label: &str,
) -> Result<Vec<u8>, Difference> {
    let len = usize::try_from(len).map_err(|_| {
        Difference::Io(format!(
            "{}: {label} length {len} does not fit this platform",
            path.display()
        ))
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|err| {
        Difference::Io(format!(
            "{}: cannot allocate {len} bytes for {label}: {err}",
            path.display()
        ))
    })?;
    bytes.resize(len, 0);
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| io_difference(path, &format!("seek to {label}"), err))?;
    file.read_exact(&mut bytes)
        .map_err(|err| io_difference(path, &format!("read {label}"), err))?;
    Ok(bytes)
}

fn masked_meta_value(bytes: &[u8], path: &Path) -> Result<serde_json::Value, Difference> {
    let meta: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|err| Difference::Io(format!("{}: meta JSON: {err}", path.display())))?;
    Ok(mask_writer_build(meta))
}

fn mask_writer_build(mut meta: serde_json::Value) -> serde_json::Value {
    if let Some(build) = meta
        .get_mut("writer")
        .and_then(|writer| writer.get_mut("build"))
    {
        *build = serde_json::Value::Null;
    }
    meta
}

fn difference_message(difference: Difference) -> String {
    match difference {
        Difference::Io(message) | Difference::Found(message) => message,
    }
}

fn ensure_same_len(hour: &OpenHour) -> Result<(), Difference> {
    let final_len = hour
        .file
        .metadata()
        .map_err(|err| io_difference(&hour.path, "re-read metadata", err))?
        .len();
    if final_len != hour.file_len {
        return Err(Difference::Io(format!(
            "{}: file length changed during comparison ({} -> {final_len})",
            hour.path.display(),
            hour.file_len
        )));
    }
    Ok(())
}

/// Compare two `.rws` hour files structurally:
///
/// - header: `version`, `index_count`
/// - meta JSON: all fields with `writer.build` masked to null
/// - index records: all fields, with `offset` normalised to payload-relative
///   (so a different-length `writer.build` that shifts absolute offsets does
///   not register as a difference)
/// - payload: byte-for-byte from each file's `payload_offset` onward
///
/// Returns `Ok(())` when equivalent.  The `Ok` path prints a summary line to
/// stdout; the `Err(Difference::Found(_))` path prints nothing — the caller
/// prints the difference.
pub fn compare(path_a: &Path, path_b: &Path) -> Result<(), Difference> {
    let mut hour_a = open_hour(path_a)?;
    let mut hour_b = open_hour(path_b)?;
    let header_a = hour_a.header;
    let header_b = hour_b.header;

    if header_a.version != header_b.version {
        return Err(Difference::Found(format!(
            "header version {} vs {}",
            header_a.version, header_b.version
        )));
    }
    if header_a.index_count != header_b.index_count {
        return Err(Difference::Found(format!(
            "index_count {} vs {}",
            header_a.index_count, header_b.index_count
        )));
    }

    // Meta JSON with writer.build masked out.
    let meta_bytes_a = read_region(
        &mut hour_a.file,
        path_a,
        HEADER_LEN as u64,
        u64::from(header_a.meta_len),
        "hour metadata",
    )?;
    let meta_bytes_b = read_region(
        &mut hour_b.file,
        path_b,
        HEADER_LEN as u64,
        u64::from(header_b.meta_len),
        "hour metadata",
    )?;
    let meta_a = masked_meta_value(&meta_bytes_a, path_a)?;
    let meta_b = masked_meta_value(&meta_bytes_b, path_b)?;
    if meta_a != meta_b {
        return Err(Difference::Found(
            "meta JSON differs beyond writer.build (variables/levels/selectors/grid_hash)"
                .to_string(),
        ));
    }

    // Index records, offsets normalized to the payload base.
    hour_a
        .file
        .seek(SeekFrom::Start(header_a.index_offset))
        .map_err(|err| io_difference(path_a, "seek to chunk index", err))?;
    hour_b
        .file
        .seek(SeekFrom::Start(header_b.index_offset))
        .map_err(|err| io_difference(path_b, "seek to chunk index", err))?;
    let mut record_bytes_a = [0u8; INDEX_RECORD_LEN];
    let mut record_bytes_b = [0u8; INDEX_RECORD_LEN];
    let mut expected_end_a = header_a.payload_offset;
    let mut expected_end_b = header_b.payload_offset;
    for index in 0..header_a.index_count {
        hour_a
            .file
            .read_exact(&mut record_bytes_a)
            .map_err(|err| io_difference(path_a, &format!("read index record {index}"), err))?;
        hour_b
            .file
            .read_exact(&mut record_bytes_b)
            .map_err(|err| io_difference(path_b, &format!("read index record {index}"), err))?;
        let record_a = ChunkRecord::unpack(&record_bytes_a).map_err(|err| {
            Difference::Io(format!("{}: index record {index}: {err}", path_a.display()))
        })?;
        let record_b = ChunkRecord::unpack(&record_bytes_b).map_err(|err| {
            Difference::Io(format!("{}: index record {index}: {err}", path_b.display()))
        })?;
        let rel_a = record_a
            .offset
            .checked_sub(header_a.payload_offset)
            .ok_or_else(|| {
                Difference::Io(format!(
                    "{}: index record {index} offset {} precedes payload offset {}",
                    path_a.display(),
                    record_a.offset,
                    header_a.payload_offset
                ))
            })?;
        let rel_b = record_b
            .offset
            .checked_sub(header_b.payload_offset)
            .ok_or_else(|| {
                Difference::Io(format!(
                    "{}: index record {index} offset {} precedes payload offset {}",
                    path_b.display(),
                    record_b.offset,
                    header_b.payload_offset
                ))
            })?;
        let end_a = record_a
            .offset
            .checked_add(u64::from(record_a.len))
            .ok_or_else(|| {
                Difference::Io(format!(
                    "{}: index record {index} payload end overflows",
                    path_a.display()
                ))
            })?;
        let end_b = record_b
            .offset
            .checked_add(u64::from(record_b.len))
            .ok_or_else(|| {
                Difference::Io(format!(
                    "{}: index record {index} payload end overflows",
                    path_b.display()
                ))
            })?;
        if end_a > hour_a.file_len || end_b > hour_b.file_len {
            return Err(Difference::Io(format!(
                "index record {index} payload exceeds file length ({} > {} or {} > {})",
                end_a, hour_a.file_len, end_b, hour_b.file_len
            )));
        }
        expected_end_a = expected_end_a.max(end_a);
        expected_end_b = expected_end_b.max(end_b);
        let fields_equal = record_a.var_id == record_b.var_id
            && record_a.kind == record_b.kind
            && record_a.flags == record_b.flags
            && record_a.tile_y == record_b.tile_y
            && record_a.tile_x == record_b.tile_x
            && rel_a == rel_b
            && record_a.len == record_b.len
            && record_a.raw_len == record_b.raw_len
            && record_a.center.to_bits() == record_b.center.to_bits()
            && record_a.scale.to_bits() == record_b.scale.to_bits()
            && record_a.min.to_bits() == record_b.min.to_bits()
            && record_a.max.to_bits() == record_b.max.to_bits()
            && record_a.valid_count == record_b.valid_count;
        if !fields_equal {
            return Err(Difference::Found(format!(
                "index record {index}: {record_a:?} (rel offset {rel_a}) vs {record_b:?} \
                 (rel offset {rel_b})"
            )));
        }
    }

    // A well-formed writer artifact ends at its last declared payload. Reject
    // sparse/trailing-byte files before streaming their potentially enormous
    // apparent payload regions.
    if hour_a.file_len != expected_end_a || hour_b.file_len != expected_end_b {
        return Err(Difference::Io(format!(
            "declared payload end does not match file length ({} vs {}, {} vs {})",
            expected_end_a, hour_a.file_len, expected_end_b, hour_b.file_len
        )));
    }

    // Payload regions, byte for byte, using fixed memory.
    let payload_len_a = hour_a.file_len - header_a.payload_offset;
    let payload_len_b = hour_b.file_len - header_b.payload_offset;
    if payload_len_a != payload_len_b {
        return Err(Difference::Found(format!(
            "payload length {} vs {}",
            payload_len_a, payload_len_b
        )));
    }
    hour_a
        .file
        .seek(SeekFrom::Start(header_a.payload_offset))
        .map_err(|err| io_difference(path_a, "seek to payload", err))?;
    hour_b
        .file
        .seek(SeekFrom::Start(header_b.payload_offset))
        .map_err(|err| io_difference(path_b, "seek to payload", err))?;
    let mut buffer_a = [0u8; COMPARE_BUFFER_BYTES];
    let mut buffer_b = [0u8; COMPARE_BUFFER_BYTES];
    let mut compared = 0u64;
    while compared < payload_len_a {
        let take = (payload_len_a - compared).min(COMPARE_BUFFER_BYTES as u64) as usize;
        hour_a
            .file
            .read_exact(&mut buffer_a[..take])
            .map_err(|err| io_difference(path_a, "read payload", err))?;
        hour_b
            .file
            .read_exact(&mut buffer_b[..take])
            .map_err(|err| io_difference(path_b, "read payload", err))?;
        if let Some(in_chunk) = buffer_a[..take]
            .iter()
            .zip(buffer_b[..take].iter())
            .position(|(a, b)| a != b)
        {
            let position = compared + in_chunk as u64;
            return Err(Difference::Found(format!(
                "payload bytes differ at payload offset {position} (of {payload_len_a})"
            )));
        }
        compared += take as u64;
    }
    ensure_same_len(&hour_a)?;
    ensure_same_len(&hour_b)?;
    println!(
        "compared: {} index records, {} payload bytes, meta keys minus writer.build",
        header_a.index_count, payload_len_a
    );
    Ok(())
}

/// Parse the meta JSON region and set `writer.build` to null so two files
/// that differ only in their build stamp compare equal.
pub fn meta_without_build(
    bytes: &[u8],
    header: &RwsHeader,
    path: &Path,
) -> Result<serde_json::Value, Difference> {
    let start = 64usize;
    let end = start + header.meta_len as usize;
    let meta: serde_json::Value =
        serde_json::from_slice(bytes.get(start..end).ok_or_else(|| {
            Difference::Io(format!("{}: meta region out of range", path.display()))
        })?)
        .map_err(|err| Difference::Io(format!("{}: meta JSON: {err}", path.display())))?;
    Ok(mask_writer_build(meta))
}

/// Read one 64-byte index record at position `index`.
pub fn record_at(
    bytes: &[u8],
    header: &RwsHeader,
    index: usize,
    path: &Path,
) -> Result<ChunkRecord, Difference> {
    let start = header.index_offset as usize + index * 64;
    let slice = bytes.get(start..start + 64).ok_or_else(|| {
        Difference::Io(format!(
            "{}: index record {index} out of range",
            path.display()
        ))
    })?;
    ChunkRecord::unpack(slice)
        .map_err(|err| Difference::Io(format!("{}: index record {index}: {err}", path.display())))
}

/// Read the `writer.build` stamp out of a store artifact: `run.json` (any
/// `.json` file with `writer.build`) or an `.rws` hour file's meta JSON.
pub fn read_writer_build(path: &Path) -> Result<String, String> {
    let is_json = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
    let mut file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|err| format!("metadata {}: {err}", path.display()))?
        .len();
    let bytes = if is_json {
        if file_len > MAX_RUN_MANIFEST_BYTES {
            return Err(format!(
                "{}: JSON is {file_len} bytes; limit is {MAX_RUN_MANIFEST_BYTES} bytes",
                path.display()
            ));
        }
        read_region(&mut file, path, 0, file_len, "JSON").map_err(difference_message)?
    } else {
        let mut header_bytes = [0u8; HEADER_LEN];
        file.read_exact(&mut header_bytes)
            .map_err(|err| format!("read header {}: {err}", path.display()))?;
        let header = RwsHeader::parse(&header_bytes)
            .map_err(|err| format!("header {}: {err}", path.display()))?;
        if u64::from(header.meta_len) > MAX_HOUR_META_BYTES {
            return Err(format!(
                "{}: hour metadata is {} bytes; limit is {MAX_HOUR_META_BYTES} bytes",
                path.display(),
                header.meta_len
            ));
        }
        if header.payload_offset > file_len {
            return Err(format!(
                "{}: index ends at {} but file length is {file_len}",
                path.display(),
                header.payload_offset
            ));
        }
        read_region(
            &mut file,
            path,
            HEADER_LEN as u64,
            u64::from(header.meta_len),
            "hour metadata",
        )
        .map_err(difference_message)?
    };
    let final_len = file
        .metadata()
        .map_err(|err| format!("re-read metadata {}: {err}", path.display()))?
        .len();
    if final_len != file_len {
        return Err(format!(
            "{}: file length changed while reading ({file_len} -> {final_len})",
            path.display()
        ));
    }
    let meta: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|err| format!("{}: JSON: {err}", path.display()))?;
    meta.get("writer")
        .and_then(|writer| writer.get("build"))
        .and_then(|build| build.as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("{}: no writer.build in meta", path.display()))
}

/// `expected` is a sha prefix (short or full). The recorded build matches
/// when it starts with the prefix and carries nothing beyond more sha hex
/// digits — so `290cf4b` matches `290cf4b2fce8` but NOT `290cf4b2fce8-dirty`:
/// a dirty build is not the claimed commit. To accept a dirty stamp
/// deliberately, pass the full stamp including the `-dirty` suffix.
pub fn build_matches(expected: &str, build: &str) -> bool {
    let Some(rest) = build.strip_prefix(expected) else {
        return false;
    };
    rest.chars().all(|c| c.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rustwx_core::{GridShape, LatLonGrid};

    use crate::ingest::HourIngestWriter;

    use super::*;

    // ── build_matches unit tests (verbatim from old bin) ──────────────────────

    #[test]
    fn build_prefix_matches_longer_sha() {
        assert!(build_matches("290cf4b", "290cf4b2fce8"));
        assert!(build_matches("290cf4b2fce8", "290cf4b2fce8"));
    }

    #[test]
    fn build_prefix_rejects_dirty_stamp() {
        assert!(!build_matches("290cf4b", "290cf4b2fce8-dirty"));
        assert!(!build_matches("290cf4b2fce8", "290cf4b2fce8-dirty"));
    }

    #[test]
    fn explicit_dirty_expectation_is_accepted() {
        assert!(build_matches("290cf4b2fce8-dirty", "290cf4b2fce8-dirty"));
    }

    #[test]
    fn build_prefix_rejects_unrelated_sha() {
        assert!(!build_matches("290cf4b", "a7bf0c7171ee"));
        assert!(!build_matches("290cf4b", "unknown"));
    }

    #[test]
    fn run_json_build_extraction() {
        let dir = std::env::temp_dir().join("rw_store_diff_test_run_json");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("run.json");
        std::fs::write(
            &path,
            r#"{"schema":"rw-store.run.v1","writer":{"name":"rw-store","version":"0.1.0","build":"290cf4b2fce8"}}"#,
        )
        .unwrap();
        assert_eq!(read_writer_build(&path).unwrap(), "290cf4b2fce8");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn oversized_json_build_source_is_rejected_before_reading() {
        let dir = test_dir("oversized-build-json");
        let path = dir.join("run.json");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_RUN_MANIFEST_BYTES + 1).unwrap();

        let error = read_writer_build(&path).unwrap_err();
        assert!(error.contains("limit"), "unexpected error: {error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_hour_meta_is_rejected_from_sparse_file() {
        let dir = test_dir("oversized-build-meta");
        let path = dir.join("f000.rws");
        let header = RwsHeader::for_layout((MAX_HOUR_META_BYTES + 1) as u32, 0);
        std::fs::write(&path, header.pack()).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(header.payload_offset)
            .unwrap();

        let error = read_writer_build(&path).unwrap_err();
        assert!(error.contains("metadata"), "unexpected error: {error}");
        assert!(error.contains("limit"), "unexpected error: {error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn excessive_index_count_is_rejected_from_header_only() {
        let dir = test_dir("excessive-index-count");
        let path = dir.join("f000.rws");
        let header = RwsHeader::for_layout(0, MAX_INDEX_RECORDS + 1);
        std::fs::write(&path, header.pack()).unwrap();

        match compare(&path, &path) {
            Err(Difference::Io(message)) => {
                assert!(
                    message.contains("index_count"),
                    "unexpected error: {message}"
                )
            }
            other => panic!("expected index-count rejection, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── compare() correctness tests ───────────────────────────────────────────

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-store-diff-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_grid() -> LatLonGrid {
        let nx = 20usize;
        let ny = 10usize;
        let lat: Vec<f32> = (0..ny)
            .flat_map(|gy| (0..nx).map(move |_gx| 30.0_f32 + 0.01 * gy as f32))
            .collect();
        let lon: Vec<f32> = (0..ny)
            .flat_map(|_gy| (0..nx).map(move |gx| -100.0_f32 + 0.05 * gx as f32))
            .collect();
        LatLonGrid::new(GridShape::new(nx, ny).unwrap(), lat, lon).unwrap()
    }

    fn write_tiny_hour(store_root: &std::path::Path, writer_build: &str) -> PathBuf {
        let grid = tiny_grid();
        let nx = 20usize;
        let ny = 10usize;
        let values: Vec<f32> = (0..ny * nx).map(|i| i as f32).collect();

        let mut writer = HourIngestWriter::begin(
            store_root,
            "test",
            "20260101_00z",
            0,
            &grid,
            None,
            writer_build,
        )
        .expect("HourIngestWriter::begin");

        writer
            .add_field_2d("t2m", "K", serde_json::json!({"var": "TMP"}), &values)
            .expect("add t2m");

        writer.finish(0).expect("finish");
        store_root.join("test").join("20260101_00z")
    }

    /// Two byte-identical files compare Ok.
    #[test]
    fn identical_files_compare_ok() {
        let dir = test_dir("identical");
        let run_dir = write_tiny_hour(&dir, "build-abc");
        let hour = run_dir.join("f000.rws");

        assert!(
            compare(&hour, &hour).is_ok(),
            "identical file should compare Ok"
        );
    }

    #[test]
    fn sparse_trailing_region_is_rejected_without_streaming_it() {
        let dir = test_dir("sparse-trailing");
        let run_dir = write_tiny_hour(&dir, "build-abc");
        let hour = run_dir.join("f000.rws");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&hour)
            .unwrap()
            .set_len(64 * 1024 * 1024)
            .unwrap();

        match compare(&hour, &hour) {
            Err(Difference::Io(message)) => assert!(
                message.contains("declared payload end"),
                "unexpected error: {message}"
            ),
            other => panic!("expected malformed sparse file, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Flipping one payload byte produces a Found difference.
    #[test]
    fn flipped_payload_byte_detected() {
        let dir = test_dir("flip");
        let run_dir = write_tiny_hour(&dir, "build-abc");
        let hour = run_dir.join("f000.rws");

        let mut bytes = std::fs::read(&hour).unwrap();
        // Parse the header to locate the payload region.
        let header = RwsHeader::parse(&bytes).unwrap();
        let payload_start = header.payload_offset as usize;
        // Flip the first payload byte.
        bytes[payload_start] ^= 0xFF;
        let corrupted = dir.join("f000_corrupt.rws");
        std::fs::write(&corrupted, &bytes).unwrap();

        match compare(&hour, &corrupted) {
            Err(Difference::Found(msg)) => {
                assert!(
                    msg.contains("payload bytes differ"),
                    "expected payload-diff message, got: {msg}"
                );
            }
            Ok(()) => panic!("expected Difference::Found, got Ok"),
            Err(Difference::Io(msg)) => panic!("expected Difference::Found, got Io: {msg}"),
        }
    }

    /// Two files differing only in writer.build still compare Ok (build masked).
    #[test]
    fn different_writer_builds_compare_ok() {
        let dir = test_dir("build-masked");
        let run_a = dir.join("store_a");
        let run_b = dir.join("store_b");
        let dir_a = write_tiny_hour(&run_a, "aaaaaaaaaaaaa");
        let dir_b = write_tiny_hour(&run_b, "bbbbbbbbbbbbb");
        let hour_a = dir_a.join("f000.rws");
        let hour_b = dir_b.join("f000.rws");

        assert!(
            compare(&hour_a, &hour_b).is_ok(),
            "files differing only in writer.build should compare Ok"
        );
    }
}
