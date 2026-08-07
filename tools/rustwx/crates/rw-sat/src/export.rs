//! PNG export of stored frames and composites, through the ported
//! production palettes — the quick-look path for validation and loops (the
//! egui viewer texture path consumes the same anchor tables).

use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};

use image::RgbaImage;

use crate::composite::{GoesAbiRgbCompositeStyle, Rgba, compose_rgb_pixels};
use crate::palette::band_color;
use crate::store::read_frame;

/// False-color one band plane. Rows are stored north-first (GOES y scan
/// axes descend), so row 0 is already the top of the image.
pub fn render_band_image(values: &[f32], nx: usize, ny: usize, band: u8) -> RgbaImage {
    let mut image = RgbaImage::new(nx as u32, ny as u32);
    for (idx, pixel) in image.pixels_mut().enumerate() {
        let color = band_color(band, values.get(idx).copied().unwrap_or(f32::NAN));
        *pixel = image::Rgba(color);
    }
    image
}

/// Compose an RGB style from same-grid band planes into an image.
pub fn render_composite_image(
    style: GoesAbiRgbCompositeStyle,
    bands: &std::collections::HashMap<u8, Vec<f32>>,
    nx: usize,
    ny: usize,
) -> Result<RgbaImage, Box<dyn Error>> {
    let pixels: Vec<Rgba> = compose_rgb_pixels(style, bands, nx * ny)?;
    let mut image = RgbaImage::new(nx as u32, ny as u32);
    for (pixel, color) in image.pixels_mut().zip(pixels) {
        *pixel = image::Rgba(color);
    }
    Ok(image)
}

/// Read a stored frame and export it as a PNG. The band comes from the
/// frame's self-describing selector. Returns the written path.
pub fn export_frame_png(
    store_root: &Path,
    model: &str,
    run: &str,
    hhmm: u16,
    out_path: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
    let frame = read_frame(store_root, model, run, hhmm)?;
    let band = frame.selector["goes"]["band"]
        .as_u64()
        .and_then(|value| u8::try_from(value).ok())
        .ok_or_else(|| {
            boxed_error(format!(
                "frame {model}/{run}/t{hhmm:04} selector carries no band"
            ))
        })?;
    let image = render_band_image(&frame.values, frame.nx, frame.ny, band);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    image.save(out_path)?;
    Ok(out_path.to_path_buf())
}

fn boxed_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::test_support::{scan_start, synthetic_field};
    use crate::store::write_band_frame;
    use std::fs;

    #[test]
    fn band_image_paints_data_and_transparent_nan() {
        let values = vec![200.0, 328.0, f32::NAN, 250.0];
        let image = render_band_image(&values, 2, 2, 13);
        assert_eq!(image.dimensions(), (2, 2));
        assert_eq!(image.get_pixel(0, 0).0[3], 255, "finite pixel is opaque");
        assert_eq!(image.get_pixel(0, 1).0, [0, 0, 0, 0], "NaN is transparent");
        // Cold (200 K) is much brighter than hot (328 K) on the IR ramp.
        assert!(image.get_pixel(0, 0).0[0] > image.get_pixel(1, 0).0[0]);
    }

    #[test]
    fn stored_frame_exports_to_png() {
        let dir = std::env::temp_dir().join(format!(
            "rw-sat-export-{}-stored-frame",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let field = synthetic_field(12, 10, scan_start(18, 51), 13, 0.0);
        let written = write_band_frame(&dir, &field, 1).unwrap();
        let png = dir.join("frame.png");
        let path = export_frame_png(&dir, &written.model, &written.run, written.hhmm, &png)
            .expect("export png");
        assert!(path.is_file());
        let loaded = image::open(&path).expect("png parses").to_rgba8();
        assert_eq!(loaded.dimensions(), (12, 10));
        let _ = fs::remove_dir_all(&dir);
    }
}
