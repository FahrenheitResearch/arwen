use crate::error::{Error, Result};
use flate2::read::ZlibDecoder;
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};

static INFLATED: AtomicU64 = AtomicU64::new(0);

/// How many DEFLATE streams this process has inflated.  A reader that
/// walks a chunked dataset should inflate each chunk it needs once; this
/// is the count that proves it (NCAR's 2.5 km restart was read with every
/// 88 MB chunk inflated once per row slab, ~60 s per variable).
pub fn inflated_streams() -> u64 {
    INFLATED.load(Ordering::Relaxed)
}

/// Decompress DEFLATE/zlib-compressed data.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    INFLATED.fetch_add(1, Ordering::Relaxed);
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
