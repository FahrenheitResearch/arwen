use crate::error::{Error, Result};
use flate2::read::ZlibDecoder;
use std::io::Read;

/// Decompress DEFLATE/zlib-compressed data.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(data);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| Error::DecompressionError(format!("DEFLATE decompression failed: {e}")))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_roundtrip() {
        let original = b"hello world, this is a test of zlib compression!";
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }
}
