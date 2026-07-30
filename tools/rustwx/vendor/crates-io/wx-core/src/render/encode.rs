//! Minimal PNG encoding for RGBA pixel buffers.
//!
//! Uses the `png` crate (already a dependency for GRIB2 Template 5.41 decoding)
//! to write RGBA images to files or byte vectors.

use std::path::Path;

/// Encode an RGBA pixel buffer as a PNG file.
///
/// # Arguments
/// * `pixels` - RGBA pixel data, row-major, 4 bytes per pixel
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `path` - Output file path
///
/// # Panics
/// Panics if `pixels.len() != width * height * 4`.
pub fn write_png<P: AsRef<Path>>(
    pixels: &[u8],
    width: u32,
    height: u32,
    path: P,
) -> Result<(), String> {
    let expected = (width as usize) * (height as usize) * 4;
    if pixels.len() != expected {
        return Err(format!(
            "Pixel buffer size {} does not match {}x{}x4 = {}",
            pixels.len(),
            width,
            height,
            expected
        ));
    }

    let file = std::fs::File::create(path.as_ref())
        .map_err(|e| format!("Failed to create file: {}", e))?;
    let buf_writer = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(buf_writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);

    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("PNG header error: {}", e))?;
    writer
        .write_image_data(pixels)
        .map_err(|e| format!("PNG write error: {}", e))?;

    Ok(())
}

/// Encode an RGBA pixel buffer as PNG bytes in memory.
///
/// Returns the PNG file as a `Vec<u8>`.
pub fn encode_png(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let expected = (width as usize) * (height as usize) * 4;
    if pixels.len() != expected {
        return Err(format!(
            "Pixel buffer size {} does not match {}x{}x4 = {}",
            pixels.len(),
            width,
            height,
            expected
        ));
    }

    let mut buf = Vec::new();

    {
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);

        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("PNG header error: {}", e))?;
        writer
            .write_image_data(pixels)
            .map_err(|e| format!("PNG write error: {}", e))?;
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_png_small() {
        // 2x2 red image
        let pixels = vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ];
        let png_bytes = encode_png(&pixels, 2, 2).unwrap();
        // PNG starts with the magic bytes
        assert_eq!(&png_bytes[..4], &[0x89, 0x50, 0x4e, 0x47]);
    }

    #[test]
    fn test_encode_png_size_mismatch() {
        let pixels = vec![0u8; 10];
        let result = encode_png(&pixels, 2, 2);
        assert!(result.is_err());
    }
}
