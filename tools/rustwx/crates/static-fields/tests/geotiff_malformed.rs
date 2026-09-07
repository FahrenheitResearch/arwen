//! GeoTIFF decode refusals for malformed geometry — audit dat-02-04, the
//! half of `bug-dat-02-02` the earlier defect pass left open.
//!
//! The module docstring of `raster::geotiff` promises to "refuse — by name —
//! anything outside that envelope rather than misread it", and it keeps that
//! promise for the type, compression and CRS enumerations.  It did not keep
//! it for the SIZES: an IFD entry count, a tag value count and the
//! tile/strip extents all came straight off disk into `usize` arithmetic and
//! a `vec![0u8; len]`.  Every case below is a file this reader accepted (or
//! panicked on) before that was fixed; each asserts a NAMED refusal, because
//! a panic inside a `cdylib` entry point takes the host process with it.
//!
//! Fixtures are built here rather than committed: they are a few dozen bytes
//! of header, and a committed corrupt raster is a file nobody can regenerate.

use static_fields::raster::{geotiff, Crs, Raster};

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("static_fields_geotiff_malformed");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// A BigTIFF header (`II`, magic 43, 8-byte offsets) whose first IFD sits at
/// byte 16, followed by that IFD's u64 entry count and nothing else.
fn bigtiff_with_entry_count(count: u64) -> Vec<u8> {
    let mut raw = Vec::new();
    raw.extend_from_slice(b"II");
    raw.extend_from_slice(&43u16.to_le_bytes());
    raw.extend_from_slice(&8u16.to_le_bytes());
    raw.extend_from_slice(&0u16.to_le_bytes());
    raw.extend_from_slice(&16u64.to_le_bytes());
    raw.extend_from_slice(&count.to_le_bytes());
    raw
}

/// Overwrite the 4-byte inline value of `tag` in a classic little-endian
/// TIFF whose IFD starts at byte 8 (which is what `write_band1` emits).
fn patch_inline_tag(raw: &mut [u8], tag: u16, value: u32) {
    let entries = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    for index in 0..entries {
        let at = 10 + index * 12;
        if u16::from_le_bytes([raw[at], raw[at + 1]]) == tag {
            raw[at + 8..at + 12].copy_from_slice(&value.to_le_bytes());
            return;
        }
    }
    panic!("tag {tag} is not in the IFD");
}

fn valid_tiff(path: &std::path::Path) {
    let (ny, nx) = (40usize, 60usize);
    let values: Vec<f64> =
        (0..ny * nx).map(|index| (index % 91) as f64 * 0.5).collect();
    let raster = Raster {
        ny,
        nx,
        values,
        transform: [0.001, 0.0, 7.25, 0.0, -0.001, 46.75],
        crs: Crs::Geographic,
    };
    geotiff::write_band1(path, &raster, geotiff::SampleType::F32, None)
        .expect("fixture writes");
}

#[test]
fn a_bigtiff_entry_count_the_file_cannot_hold_is_refused_by_name() {
    // 2^61 entries * 20 bytes each overflows usize before anything is
    // allocated: a debug panic, and in release a wrapped length handed
    // straight to `vec![0u8; len]`.  The 24-byte file is the ceiling and
    // the refusal says so.  (The merely-enormous non-overflowing value is
    // deliberately not exercised here: proving it would ask this test's own
    // process for tens of gigabytes.)
    let path = scratch("bigtiff_entry_count.tif");
    std::fs::write(&path, bigtiff_with_entry_count(1u64 << 61)).unwrap();
    let err = match geotiff::TiffReader::open(&path) {
        Ok(_) => panic!("malformed file was accepted"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("IFD entry table"), "{err}");
    assert!(err.contains("cannot hold"), "{err}");
}

#[test]
fn a_bigtiff_tag_count_the_file_cannot_hold_is_refused_by_name() {
    // One well-formed IFD entry declaring 2^62 DOUBLEs: 8 * 2^62 overflows
    // u64, so the payload length is computed, not read, into nonsense.
    let mut raw = bigtiff_with_entry_count(1);
    raw.extend_from_slice(&256u16.to_le_bytes()); // ImageWidth
    raw.extend_from_slice(&12u16.to_le_bytes()); // DOUBLE, 8 bytes
    raw.extend_from_slice(&(1u64 << 62).to_le_bytes());
    raw.extend_from_slice(&64u64.to_le_bytes()); // payload offset
    raw.extend_from_slice(&0u64.to_le_bytes()); // no next IFD
    let path = scratch("bigtiff_tag_count.tif");
    std::fs::write(&path, raw).unwrap();
    let err = match geotiff::TiffReader::open(&path) {
        Ok(_) => panic!("malformed file was accepted"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("tag payload"), "{err}");
    assert!(err.contains("cannot hold"), "{err}");
}

#[test]
fn a_zero_pixel_image_is_refused_at_open() {
    let path = scratch("zero_width.tif");
    valid_tiff(&path);
    let mut raw = std::fs::read(&path).unwrap();
    patch_inline_tag(&mut raw, 256, 0); // ImageWidth
    std::fs::write(&path, &raw).unwrap();
    let err = match geotiff::TiffReader::open(&path) {
        Ok(_) => panic!("malformed file was accepted"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("0x40 image"), "{err}");
}

#[test]
fn a_zero_extent_block_is_refused_at_open_not_divided_by_later() {
    // The concrete case in the finding: a truncated download leaves a
    // plausible header and a `00 00` block-length word.  `open()` used to
    // accept it and `read_window_raw`'s `row_off / self.block_h` then
    // divided by zero -- a process abort, not a refusal.
    let path = scratch("zero_tile_length.tif");
    valid_tiff(&path);
    let mut raw = std::fs::read(&path).unwrap();
    patch_inline_tag(&mut raw, 323, 0); // TileLength
    std::fs::write(&path, &raw).unwrap();
    let err = match geotiff::TiffReader::open(&path) {
        Ok(_) => panic!("malformed file was accepted"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("block"), "{err}");
    assert!(err.contains("must be positive"), "{err}");
}

#[test]
fn an_empty_window_is_an_empty_answer_and_not_an_underflow() {
    // `block_row_hi = (row_off + win_h - 1) / block_h` underflows usize for
    // win_h == 0: a debug panic, and in release a 0..=usize::MAX block loop.
    let path = scratch("empty_window.tif");
    valid_tiff(&path);
    let mut reader = geotiff::TiffReader::open(&path).expect("opens");
    assert!(reader.read_window_raw(0, 0, 0, 0).expect("empty").is_empty());
    assert!(reader.read_window_raw(3, 5, 0, 7).expect("empty").is_empty());
    assert!(reader.read_window_raw(3, 5, 7, 0).expect("empty").is_empty());
    // A one-pixel window still decodes, so the guard is not swallowing work.
    assert_eq!(reader.read_window_raw(0, 0, 1, 1).expect("one").len(), 1);
}
