//! HDF5 LZ4 filter (filter ID 32004).
//!
//! On-disk format per the [HDF5 LZ4 plugin](https://github.com/HDFGroup/hdf5_plugins):
//! - `orig_size: u64 BE` — total uncompressed size of the chunk
//! - Repeated blocks:
//!   - `orig_block_size: u32 BE` — uncompressed size of this block
//!   - `compressed_size: u32 BE` — compressed size of this block
//!   - `compressed_data: [u8; compressed_size]`
//!
//! When `compressed_size == orig_block_size` the block is stored uncompressed.

use crate::error::{Error, Result};

/// Decompress HDF5 LZ4-filtered data.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 8 {
        return Err(Error::DecompressionError(
            "LZ4: input too short for header".into(),
        ));
    }

    let orig_size = u64::from_be_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;
    let mut output = Vec::with_capacity(orig_size);
    let mut pos = 8;

    while pos < data.len() && output.len() < orig_size {
        if pos + 8 > data.len() {
            return Err(Error::DecompressionError(
                "LZ4: truncated block header".into(),
            ));
        }

        let orig_block_size =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        let compressed_size =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + compressed_size > data.len() {
            return Err(Error::DecompressionError(
                "LZ4: truncated block data".into(),
            ));
        }

        if compressed_size == orig_block_size {
            // Stored uncompressed
            output.extend_from_slice(&data[pos..pos + compressed_size]);
        } else {
            let decompressed =
                lz4_flex::decompress(&data[pos..pos + compressed_size], orig_block_size)
                    .map_err(|e| Error::DecompressionError(format!("LZ4: {}", e)))?;
            output.extend_from_slice(&decompressed);
        }
        pos += compressed_size;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_round_trip() {
        let original = b"hello world hello world hello world!";
        let raw_compressed = lz4_flex::compress(original);

        let mut hdf5_data = Vec::new();
        hdf5_data.extend_from_slice(&(original.len() as u64).to_be_bytes());
        // Block header: orig_block_size + compressed_size
        hdf5_data.extend_from_slice(&(original.len() as u32).to_be_bytes());
        hdf5_data.extend_from_slice(&(raw_compressed.len() as u32).to_be_bytes());
        hdf5_data.extend_from_slice(&raw_compressed);

        let result = decompress(&hdf5_data).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn test_lz4_uncompressed_block() {
        let original = b"short";
        let mut hdf5_data = Vec::new();
        hdf5_data.extend_from_slice(&(original.len() as u64).to_be_bytes());
        // compressed_size == orig_block_size => uncompressed
        hdf5_data.extend_from_slice(&(original.len() as u32).to_be_bytes());
        hdf5_data.extend_from_slice(&(original.len() as u32).to_be_bytes());
        hdf5_data.extend_from_slice(original);

        let result = decompress(&hdf5_data).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn test_lz4_too_short() {
        let data = &[0; 7];
        assert!(decompress(data).is_err());
    }

    #[test]
    fn test_lz4_truncated_block() {
        let mut data = vec![0, 0, 0, 0, 0, 0, 0, 10]; // orig_size = 10
        data.extend_from_slice(&10u32.to_be_bytes()); // orig_block_size = 10
        data.extend_from_slice(&100u32.to_be_bytes()); // compressed_size = 100
        data.extend_from_slice(&[0; 5]); // only 5 bytes of "100"
        assert!(decompress(&data).is_err());
    }
}
