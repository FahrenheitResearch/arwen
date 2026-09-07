//! What the GeoTIFF reader accepts, and what it must refuse by name.
//!
//! `TiffReader::open` accepts both TIFF byte orders (`II` and `MM`), so the
//! decode envelope the module docstring advertises includes big-endian
//! rasters.  These build the smallest file inside that envelope -- one
//! uncompressed 3x2 `int16` strip with the horizontal predictor -- byte by
//! byte, because no writer in this tree emits one.

use std::path::Path;

use static_fields::raster::geotiff;

const MODEL_TRANSFORMATION: u16 = 34264;

/// One IFD entry: tag, field type, value count, and its 4-byte value field.
struct Entry {
    tag: u16,
    field_type: u16,
    count: u32,
    value: [u8; 4],
}

fn short(tag: u16, value: u16) -> Entry {
    // Inline values are left-justified in the 4-byte field.
    let [high, low] = value.to_be_bytes();
    Entry { tag, field_type: 3, count: 1, value: [high, low, 0, 0] }
}

fn long(tag: u16, value: u32) -> Entry {
    Entry { tag, field_type: 4, count: 1, value: value.to_be_bytes() }
}

/// A minimal big-endian GeoTIFF: single uncompressed strip, one `int16`
/// band, `Predictor = 2`, and a `ModelTransformation` so the georeferencing
/// gate is satisfied.  `deltas` are the stored (differenced) samples.
fn big_endian_predictor2_tiff(deltas: &[i16], nx: usize, ny: usize) -> Vec<u8> {
    assert_eq!(deltas.len(), nx * ny);
    let entries = vec![
        short(256, nx as u16),
        short(257, ny as u16),
        short(258, 16),
        short(259, 1),
        long(273, 0),           // StripOffsets, patched below
        short(277, 1),
        short(278, ny as u16),
        long(279, (deltas.len() * 2) as u32),
        short(317, 2),          // Predictor: horizontal differencing
        short(339, 2),          // SampleFormat: signed integer
        Entry { tag: MODEL_TRANSFORMATION, field_type: 12, count: 16,
                value: [0, 0, 0, 0] },  // offset, patched below
    ];

    let ifd_offset = 8u32;
    let ifd_len = 2 + entries.len() as u32 * 12 + 4;
    let matrix_offset = ifd_offset + ifd_len;
    let strip_offset = matrix_offset + 16 * 8;

    let mut out = Vec::new();
    out.extend_from_slice(b"MM");
    out.extend_from_slice(&42u16.to_be_bytes());
    out.extend_from_slice(&ifd_offset.to_be_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_be_bytes());
    for entry in &entries {
        out.extend_from_slice(&entry.tag.to_be_bytes());
        out.extend_from_slice(&entry.field_type.to_be_bytes());
        out.extend_from_slice(&entry.count.to_be_bytes());
        match entry.tag {
            273 => out.extend_from_slice(&strip_offset.to_be_bytes()),
            MODEL_TRANSFORMATION => {
                out.extend_from_slice(&matrix_offset.to_be_bytes())
            }
            _ => out.extend_from_slice(&entry.value),
        }
    }
    out.extend_from_slice(&0u32.to_be_bytes());  // no second IFD
    // ModelTransformation: 1 degree pixels with the origin at (0, 0).
    let matrix = [
        1.0f64, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    for value in matrix {
        out.extend_from_slice(&value.to_be_bytes());
    }
    assert_eq!(out.len(), strip_offset as usize);
    for delta in deltas {
        out.extend_from_slice(&delta.to_be_bytes());
    }
    out
}

fn read_all(path: &Path) -> Vec<f64> {
    let mut reader = geotiff::TiffReader::open(path).expect("the tile opens");
    let (nx, ny) = (reader.width, reader.height);
    reader.read_window_raw(0, 0, nx, ny).expect("the window reads")
}

#[test]
fn a_big_endian_horizontal_predictor_decodes_to_the_encoded_values() {
    // Row 0 stores deltas 200, 100, -200, i.e. the true row [200, 300, 100].
    // Undone on little-endian words the carries cross the wrong byte lanes:
    // 200 survives and 300 comes back as 44 -- a 300 m ridge published as a
    // 44 m one, with no error and no NaN.
    let bytes = big_endian_predictor2_tiff(&[200, 100, -200, -5, 10, -5], 3, 2);
    let directory = std::env::temp_dir()
        .join(format!("gpuwm-geotiff-be-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("big_endian_predictor2.tif");
    std::fs::write(&path, &bytes).unwrap();

    let values = read_all(&path);
    std::fs::remove_dir_all(&directory).ok();

    assert_eq!(values, vec![200.0, 300.0, 100.0, -5.0, 5.0, 0.0]);
}

#[test]
fn an_ifd_field_type_outside_the_table_is_refused_not_indexed() {
    // Negative control for the tag reader's table lookup.  `field_type` is a
    // raw u16 off disk and `TYPE_SIZES` holds 19 entries, so a corrupt or
    // private type used to index past the array -- a panic that, at the
    // cdylib seam, aborted the host Python interpreter.
    let mut bytes = big_endian_predictor2_tiff(&[200, 100, -200, -5, 10, -5], 3, 2);
    let entry = 8 + 2;  // the first IFD entry, which is ImageWidth
    assert_eq!(u16::from_be_bytes([bytes[entry], bytes[entry + 1]]), 256);
    bytes[entry + 2..entry + 4].copy_from_slice(&0x2000u16.to_be_bytes());

    let directory = std::env::temp_dir()
        .join(format!("gpuwm-geotiff-type-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("unknown_field_type.tif");
    std::fs::write(&path, &bytes).unwrap();

    let opened = geotiff::TiffReader::open(&path);
    std::fs::remove_dir_all(&directory).ok();

    let message = opened.err().expect("an unreadable IFD is refused").to_string();
    assert!(message.contains("ImageWidth"), "{message}");
}
