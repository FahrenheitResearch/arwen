//! A Blosc (format 2, the c-blosc 1.x frame) encoder: byte shuffle, then
//! Zstandard per block.
//!
//! Why Blosc and not a bare Zstandard codec: the Zarr format 2 compressor
//! `{"id": "blosc", "cname": "zstd", "shuffle": 1}` is self-contained (the
//! shuffle travels inside the frame, so the array needs no filter chain),
//! and it is the compressor Zarr 2 readers have carried longest; numcodecs
//! decodes it under zarr-python 2 and 3, which is what xarray reads with.
//!
//! The frame, as c-blosc's `blosc_d` reads it:
//!
//! ```text
//! 0  version (2)          1  codec format version (zstd: 1)
//! 2  flags                3  typesize
//! 4  nbytes (u32 LE)      8  blocksize (u32 LE)     12  cbytes (u32 LE)
//! 16 bstarts: one u32 per block, the offset of its stream
//!    then per block: i32 compressed length, then the stream
//! ```
//!
//! Flags: 0x01 byte shuffle, 0x02 stored without compression (the payload
//! follows the 16-byte header), 0x10 the block is not split per byte
//! position, bits 5-7 the codec (zstd is 4).  A block whose compressed
//! length would reach its raw length is stored raw, which `blosc_d` knows by
//! the length being equal.
//!
//! Deterministic: the same bytes in give the same frame out, whatever the
//! thread count, because blocks are compressed independently and assembled
//! in order.

use rayon::prelude::*;

/// Block size in bytes: a multiple of every element size used here.
pub const BLOCK_BYTES: usize = 256 * 1024;
const HEADER: usize = 16;
const VERSION: u8 = 2;
const ZSTD_FORMAT_VERSION: u8 = 1;
const FLAG_SHUFFLE: u8 = 0x01;
const FLAG_MEMCPYED: u8 = 0x02;
const FLAG_DONT_SPLIT: u8 = 0x10;
const CODEC_ZSTD: u8 = 4 << 5;

/// The `.zarray` compressor entry for frames from [`compress`].
pub fn zarr_compressor(clevel: i32) -> serde_json::Value {
    serde_json::json!({
        "id": "blosc",
        "cname": "zstd",
        "clevel": clevel,
        "shuffle": 1,
        "blocksize": 0,
    })
}

fn shuffle(block: &[u8], typesize: usize, out: &mut Vec<u8>) {
    out.clear();
    out.resize(block.len(), 0);
    let n = block.len() / typesize;
    for (i, element) in block[..n * typesize].chunks_exact(typesize).enumerate() {
        for (j, &byte) in element.iter().enumerate() {
            out[j * n + i] = byte;
        }
    }
    // A trailing partial element is copied as it is (c-blosc does the same).
    out[n * typesize..].copy_from_slice(&block[n * typesize..]);
}

/// Compress `data` (elements of `typesize` bytes) into one Blosc frame.
pub fn compress(data: &[u8], typesize: usize, clevel: i32) -> std::io::Result<Vec<u8>> {
    assert!(typesize >= 1 && typesize <= 255);
    let nbytes = data.len();
    if nbytes > (u32::MAX as usize) - HEADER {
        return Err(std::io::Error::other(
            "a Blosc frame holds at most 4 GiB; the chunk planner keeps chunks under 64 MiB",
        ));
    }
    let blocksize = if nbytes == 0 { 0 } else { BLOCK_BYTES.min(nbytes) };
    let shuffled = typesize > 1;
    let mut flags = FLAG_DONT_SPLIT | CODEC_ZSTD;
    if shuffled {
        flags |= FLAG_SHUFFLE;
    }
    let streams: Vec<Vec<u8>> = if nbytes == 0 {
        Vec::new()
    } else {
        data.par_chunks(blocksize)
            .map(|block| -> std::io::Result<Vec<u8>> {
                let mut scratch = Vec::new();
                let source: &[u8] = if shuffled {
                    shuffle(block, typesize, &mut scratch);
                    &scratch
                } else {
                    block
                };
                let packed = zstd::bulk::compress(source, clevel)?;
                let mut stream = Vec::with_capacity(4 + packed.len().min(source.len()));
                if packed.len() >= source.len() {
                    stream.extend_from_slice(&(source.len() as i32).to_le_bytes());
                    stream.extend_from_slice(source);
                } else {
                    stream.extend_from_slice(&(packed.len() as i32).to_le_bytes());
                    stream.extend_from_slice(&packed);
                }
                Ok(stream)
            })
            .collect::<std::io::Result<Vec<_>>>()?
    };
    let nblocks = streams.len();
    let total: usize = HEADER + 4 * nblocks + streams.iter().map(Vec::len).sum::<usize>();
    if total >= nbytes + HEADER {
        // Not worth compressing: c-blosc's memcpyed frame.
        let mut frame = Vec::with_capacity(nbytes + HEADER);
        frame.extend_from_slice(&[
            VERSION,
            ZSTD_FORMAT_VERSION,
            FLAG_MEMCPYED | FLAG_DONT_SPLIT | CODEC_ZSTD,
            typesize as u8,
        ]);
        frame.extend_from_slice(&(nbytes as u32).to_le_bytes());
        frame.extend_from_slice(&(blocksize as u32).to_le_bytes());
        frame.extend_from_slice(&((nbytes + HEADER) as u32).to_le_bytes());
        frame.extend_from_slice(data);
        return Ok(frame);
    }
    let mut frame = Vec::with_capacity(total);
    frame.extend_from_slice(&[VERSION, ZSTD_FORMAT_VERSION, flags, typesize as u8]);
    frame.extend_from_slice(&(nbytes as u32).to_le_bytes());
    frame.extend_from_slice(&(blocksize as u32).to_le_bytes());
    frame.extend_from_slice(&(total as u32).to_le_bytes());
    let mut offset = HEADER + 4 * nblocks;
    for stream in &streams {
        frame.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += stream.len();
    }
    for stream in &streams {
        frame.extend_from_slice(stream);
    }
    debug_assert_eq!(frame.len(), total);
    Ok(frame)
}

/// Decode a frame from [`compress`].  Test and self-check use only: the
/// datasets are read by numcodecs, and the independent proof of the format
/// is numcodecs decoding these frames (`tests/test_ml_export.py`).
pub fn decompress(frame: &[u8]) -> std::io::Result<Vec<u8>> {
    let bad = |what: &str| std::io::Error::other(format!("bad Blosc frame: {what}"));
    if frame.len() < HEADER || frame[0] != VERSION {
        return Err(bad("header"));
    }
    let flags = frame[2];
    let typesize = frame[3] as usize;
    let word = |at: usize| u32::from_le_bytes(frame[at..at + 4].try_into().unwrap()) as usize;
    let (nbytes, blocksize, cbytes) = (word(4), word(8), word(12));
    if cbytes != frame.len() {
        return Err(bad("length"));
    }
    if flags & FLAG_MEMCPYED != 0 {
        return Ok(frame[HEADER..HEADER + nbytes].to_vec());
    }
    let nblocks = if blocksize == 0 { 0 } else { nbytes.div_ceil(blocksize) };
    let mut out = Vec::with_capacity(nbytes);
    for b in 0..nblocks {
        let start = word(HEADER + 4 * b);
        let neblock = blocksize.min(nbytes - b * blocksize);
        let clen = i32::from_le_bytes(frame[start..start + 4].try_into().unwrap()) as usize;
        let payload = &frame[start + 4..start + 4 + clen];
        let raw = if clen == neblock {
            payload.to_vec()
        } else {
            zstd::bulk::decompress(payload, neblock)?
        };
        if raw.len() != neblock {
            return Err(bad("block length"));
        }
        if flags & FLAG_SHUFFLE != 0 && typesize > 1 {
            let n = neblock / typesize;
            let base = out.len();
            out.resize(base + neblock, 0);
            for i in 0..n {
                for j in 0..typesize {
                    out[base + i * typesize + j] = raw[j * n + i];
                }
            }
            out[base + n * typesize..base + neblock].copy_from_slice(&raw[n * typesize..]);
        } else {
            out.extend_from_slice(&raw);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floats(n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|i| (250.0f32 + (i as f32 * 0.01).sin() * 30.0).to_le_bytes())
            .collect()
    }

    #[test]
    fn round_trips_multi_block_and_leftover() {
        let data = floats(BLOCK_BYTES / 4 * 3 + 17);
        let frame = compress(&data, 4, 3).unwrap();
        assert!(frame.len() < data.len());
        assert_eq!(frame[2], FLAG_SHUFFLE | FLAG_DONT_SPLIT | CODEC_ZSTD);
        assert_eq!(decompress(&frame).unwrap(), data);
    }

    #[test]
    fn header_says_what_blosc_d_reads() {
        let data = floats(1000);
        let frame = compress(&data, 4, 3).unwrap();
        assert_eq!(&frame[0..2], &[2, 1]);
        assert_eq!(frame[3], 4);
        assert_eq!(u32::from_le_bytes(frame[4..8].try_into().unwrap()), 4000);
        assert_eq!(u32::from_le_bytes(frame[8..12].try_into().unwrap()), 4000);
        assert_eq!(u32::from_le_bytes(frame[12..16].try_into().unwrap()) as usize, frame.len());
        assert_eq!(u32::from_le_bytes(frame[16..20].try_into().unwrap()), 20);
    }

    #[test]
    fn incompressible_bytes_become_a_memcpyed_frame() {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let data: Vec<u8> = (0..4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        let frame = compress(&data, 1, 3).unwrap();
        assert_eq!(frame[2] & FLAG_MEMCPYED, FLAG_MEMCPYED);
        assert_eq!(frame.len(), data.len() + 16);
        assert_eq!(decompress(&frame).unwrap(), data);
    }

    #[test]
    fn one_byte_elements_are_not_shuffled() {
        let data = vec![0u8, 1, 1, 0, 1, 1, 1, 0].repeat(512);
        let frame = compress(&data, 1, 3).unwrap();
        assert_eq!(frame[2] & FLAG_SHUFFLE, 0);
        assert_eq!(decompress(&frame).unwrap(), data);
    }

    #[test]
    fn frames_are_deterministic_across_thread_counts() {
        let data = floats(BLOCK_BYTES / 2 + 1000);
        let a = compress(&data, 4, 3).unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let b = pool.install(|| compress(&data, 4, 3).unwrap());
        assert_eq!(a, b);
    }
}
