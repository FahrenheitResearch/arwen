use crate::codecs::blosclz;
use crate::filters;
use crate::internal::constants::*;
use std::io::Write;

pub mod constants;

/// Convert compressor code to compressor format (for header flags byte).
///
/// Blosc1 headers store `BLOSC_SNAPPY_LIB` (2) for snappy, while blosc2 extended
/// headers store the raw compcode in byte 22 and the format field is less critical.
/// Pass `is_blosc1 = true` when writing a 16-byte blosc1 header.
fn compcode_to_compformat(compcode: u8, is_blosc1: bool) -> u8 {
    match compcode {
        BLOSC_BLOSCLZ => BLOSC_BLOSCLZ_FORMAT,
        BLOSC_LZ4 => BLOSC_LZ4_FORMAT,
        BLOSC_LZ4HC => BLOSC_LZ4HC_FORMAT,
        BLOSC_SNAPPY => {
            if is_blosc1 {
                BLOSC_SNAPPY_FORMAT // 2 (BLOSC_SNAPPY_LIB in blosc1)
            } else {
                BLOSC_SNAPPY // 3 (no separate lib code in blosc2)
            }
        }
        BLOSC_ZLIB => BLOSC_ZLIB_FORMAT,
        BLOSC_ZSTD => BLOSC_ZSTD_FORMAT,
        _ => compcode,
    }
}

/// Convert compressor format (from header flags byte) back to compressor code.
///
/// Blosc1 headers store the compressor *library* code directly in flags bits 5–7
/// (BLOSC_SNAPPY_LIB=2, BLOSC_ZLIB_LIB=3, …).  Blosc2 extended headers store the
/// same field but disambiguate via the explicit compcode in byte 22, so this
/// function is only needed for the blosc1 path.
fn compformat_to_compcode(compformat: u8, is_blosc1: bool) -> u8 {
    if is_blosc1 {
        match compformat {
            BLOSC_BLOSCLZ_FORMAT => BLOSC_BLOSCLZ, // 0 → 0
            BLOSC_LZ4_FORMAT => BLOSC_LZ4,         // 1 → 1 (also LZ4HC)
            BLOSC_SNAPPY_FORMAT => BLOSC_SNAPPY,   // 2 → 3 (blosc1: SNAPPY_LIB=2)
            BLOSC_ZLIB_FORMAT => BLOSC_ZLIB,       // 3 → 4
            BLOSC_ZSTD_FORMAT => BLOSC_ZSTD,       // 4 → 5
            _ => compformat,
        }
    } else {
        // Blosc2 non-extended headers are not expected in practice (byte 22 is
        // authoritative), but provide a best-effort mapping as a fallback.
        match compformat {
            BLOSC_BLOSCLZ_FORMAT => BLOSC_BLOSCLZ,
            BLOSC_LZ4_FORMAT => BLOSC_LZ4,
            BLOSC_ZLIB_FORMAT => BLOSC_ZLIB,
            BLOSC_ZSTD_FORMAT => BLOSC_ZSTD,
            _ => compformat,
        }
    }
}

fn create_header_blosc1(
    nbytes: usize,
    blocksize: usize,
    cbytes: usize,
    typesize: usize,
    flags: u8,
    compressor: u8,
) -> [u8; BLOSC_MIN_HEADER_LENGTH] {
    let mut header = [0u8; BLOSC_MIN_HEADER_LENGTH];
    header[0] = BLOSC1_VERSION_FORMAT;
    header[1] = 1; // Version for compressor (e.g., 1 for blosclz)
    header[2] = flags | (compcode_to_compformat(compressor, true) << 5);
    header[3] = typesize as u8;
    header[4..8].copy_from_slice(&(nbytes as u32).to_le_bytes());
    header[8..12].copy_from_slice(&(blocksize as u32).to_le_bytes());
    header[12..16].copy_from_slice(&(cbytes as u32).to_le_bytes());
    header
}

fn create_header_blosc2(
    nbytes: usize,
    blocksize: usize,
    cbytes: usize,
    typesize: usize,
    flags: u8,
    compressor: u8,
    filters: &[u8; 6],
    filters_meta: &[u8; 6],
) -> [u8; BLOSC_EXTENDED_HEADER_LENGTH] {
    let mut header = [0u8; BLOSC_EXTENDED_HEADER_LENGTH];
    // First 16 bytes: standard Blosc header
    header[0] = BLOSC2_VERSION_FORMAT_STABLE;
    header[1] = 1; // Version for compressor (e.g., 1 for blosclz)
    header[2] = flags | (compcode_to_compformat(compressor, false) << 5);
    header[3] = typesize as u8;
    header[4..8].copy_from_slice(&(nbytes as u32).to_le_bytes());
    header[8..12].copy_from_slice(&(blocksize as u32).to_le_bytes());
    header[12..16].copy_from_slice(&(cbytes as u32).to_le_bytes());

    // Extended header (bytes 16-31)
    // Bytes 16-21: filters[0..6]
    header[16..22].copy_from_slice(filters);
    // Byte 22: compressor code (same as bits 5-7 of flags byte)
    header[22] = compressor;
    // Byte 23: compressor metadata
    header[23] = 0;
    // Bytes 24-29: filters_meta[0..6]
    header[24..30].copy_from_slice(filters_meta);
    // Byte 30: reserved
    header[30] = 0;
    // Byte 31: blosc2 flags
    header[31] = 0;

    header
}

/// Compresses `src` into `dest` using the given codec and a default filter pipeline.
///
/// This is the lower-level entry point used by [`crate::blosc2_compress`]. It handles
/// block sizing, header construction, filter application, and per-block codec invocation.
///
/// Returns the total number of bytes written to `dest` (including the header).
pub fn compress(
    clevel: i32,
    doshuffle: i32,
    typesize: usize,
    src: &[u8],
    dest: &mut [u8],
    compressor: u8,
) -> Result<usize, i32> {
    let mut filters = [BLOSC_NOFILTER; 6];
    if doshuffle == BLOSC_SHUFFLE as i32 && typesize > 1 {
        filters[5] = BLOSC_SHUFFLE;
    } else if doshuffle == BLOSC_BITSHUFFLE as i32 {
        filters[5] = BLOSC_BITSHUFFLE;
    }

    compress_internal(
        clevel,
        doshuffle,
        typesize,
        src,
        dest,
        compressor,
        false,
        &filters,
        &[0; 6],
    )
}

/// Like [`compress`], but accepts an explicit filter pipeline and filter metadata.
///
/// Used by [`crate::blosc2_compress_ctx`] to support custom filter configurations
/// beyond the simple shuffle modes.
pub fn compress_extended(
    clevel: i32,
    doshuffle: i32,
    typesize: usize,
    src: &[u8],
    dest: &mut [u8],
    compressor: u8,
    filters: &[u8; 6],
    filters_meta: &[u8; 6],
) -> Result<usize, i32> {
    compress_internal(
        clevel,
        doshuffle,
        typesize,
        src,
        dest,
        compressor,
        true,
        filters,
        filters_meta,
    )
}

fn compute_blocksize(
    clevel: i32,
    typesize: usize,
    nbytes: usize,
    compressor: u8,
    filter_flags: u8,
    extended_header: bool,
) -> usize {
    if nbytes < typesize {
        return nbytes.max(1);
    }

    let mut blocksize = nbytes;

    // Check splitmode using the initial blocksize (= nbytes), matching C behavior
    let splitmode = split_block(compressor, clevel, typesize, blocksize, filter_flags, extended_header);

    if nbytes >= L1 {
        blocksize = L1;

        let is_hcr = match compressor {
            BLOSC_LZ4HC | BLOSC_ZLIB | BLOSC_ZSTD => true,
            _ => false,
        };

        if is_hcr {
            blocksize *= 2;
        }

        match clevel {
            0 => blocksize /= 4,
            1 => blocksize /= 2,
            2 => blocksize *= 1,
            3 => blocksize *= 2,
            4 | 5 => blocksize *= 4,
            6 | 7 | 8 => blocksize *= 8,
            9 => {
                blocksize *= 8;
                if is_hcr {
                    blocksize *= 2;
                }
            }
            _ => {}
        }
    }

    // Override blocksize for splittable codecs (matching C stune.c)
    if clevel > 0 && splitmode {
        blocksize = match clevel {
            1 | 2 | 3 => 32 * 1024,
            4 | 5 | 6 => 64 * 1024,
            7 => 128 * 1024,
            8 => 256 * 1024,
            _ => 512 * 1024, // clevel 9 and above
        };
        // Multiply by typesize to get proper split sizes
        blocksize *= typesize;
        // But do not exceed 4 MB
        if blocksize > 4 * 1024 * 1024 {
            blocksize = 4 * 1024 * 1024;
        }
        if blocksize < 32 * 1024 {
            // Do not use a too small blocksize (< 32 KB) when typesize is small
            blocksize = 32 * 1024;
        }
    }

    if blocksize > nbytes {
        blocksize = nbytes;
    }

    // blocksize must be a multiple of typesize (matching C's stune.c)
    if typesize > 0 && blocksize > typesize {
        blocksize = (blocksize / typesize) * typesize;
    }

    blocksize
}

/// Compute filter_flags from the filters array, matching C's filters_to_flags().
fn filters_to_flags(filters: &[u8; 6]) -> u8 {
    let mut flags = 0u8;
    for &f in filters.iter() {
        match f {
            BLOSC_SHUFFLE => flags |= BLOSC_DOSHUFFLE,
            BLOSC_BITSHUFFLE => flags |= BLOSC_DOBITSHUFFLE,
            _ => {}
        }
    }
    flags
}

/// Blosc2 split logic (c-blosc2 stune.c): only split for byte shuffle, not bitshuffle.
fn split_block_blosc2(
    compressor: u8,
    clevel: i32,
    typesize: usize,
    blocksize: usize,
    filter_flags: u8,
) -> bool {
    if (filter_flags & BLOSC_DOSHUFFLE) == 0 {
        return false;
    }

    let split = match compressor {
        BLOSC_BLOSCLZ | BLOSC_LZ4 => true,
        BLOSC_ZSTD if clevel <= 5 => true,
        _ => false,
    };

    split && (typesize <= 16) && (blocksize / typesize >= BLOSC_MIN_BUFFERSIZE)
}

/// Blosc1 split logic (c-blosc blosc.c): FORWARD_COMPAT_SPLIT mode, no shuffle check.
fn split_block_blosc1(
    compressor: u8,
    typesize: usize,
    blocksize: usize,
) -> bool {
    (compressor != BLOSC_ZSTD)
        && (typesize <= 16)
        && (blocksize / typesize >= BLOSC_MIN_BUFFERSIZE)
}

fn split_block(
    compressor: u8,
    clevel: i32,
    typesize: usize,
    blocksize: usize,
    filter_flags: u8,
    extended_header: bool,
) -> bool {
    if extended_header {
        split_block_blosc2(compressor, clevel, typesize, blocksize, filter_flags)
    } else {
        split_block_blosc1(compressor, typesize, blocksize)
    }
}

fn compress_internal(
    clevel: i32,
    doshuffle: i32,
    typesize: usize,
    src: &[u8],
    dest: &mut [u8],
    compressor: u8,
    extended_header: bool,
    filters: &[u8; 6],
    filters_meta: &[u8; 6],
) -> Result<usize, i32> {
    let nbytes = src.len();

    // Compute actual filter flags from the filters array (matching C's filters_to_flags)
    // Must be computed before blocksize since split_block depends on it.
    let filter_flags = filters_to_flags(filters);

    let blocksize = compute_blocksize(clevel, typesize, nbytes, compressor, filter_flags, extended_header);
    let nblocks = if nbytes == 0 {
        0
    } else {
        (nbytes + blocksize - 1) / blocksize
    };

    let header_len = if extended_header {
        BLOSC_EXTENDED_HEADER_LENGTH
    } else {
        BLOSC_MIN_HEADER_LENGTH
    };

    // Header flags: for extended header, both DOSHUFFLE and DOBITSHUFFLE are set as a marker
    let mut flags = filter_flags;
    if extended_header {
        flags |= BLOSC_DOSHUFFLE | BLOSC_DOBITSHUFFLE;
    }

    // Use actual filter_flags (not header flags) for split decision, matching C behavior
    let split = split_block(compressor, clevel, typesize, blocksize, filter_flags, extended_header);

    if !split {
        // For blosc2, only set dont_split when clevel > 0 (matching C blosc2 stune.c)
        // For blosc1, always set dont_split when not splitting (matching C blosc1 write_compression_header)
        if !extended_header || (clevel != 0 && nbytes >= BLOSC_MIN_BUFFERSIZE) {
            flags |= 0x10;
        }
    }

    // Calculate data start offset
    let mut data_offset = header_len;
    // Always add bstarts if nblocks > 0 (Blosc1 behavior)
    if nblocks > 0 {
        data_offset += nblocks * 4;
    }

    let mut current_dest_offset = data_offset;
    let mut incompressible = false;

    let mut bstarts = vec![0usize; nblocks];

    for i in 0..nblocks {
        let start = i * blocksize;
        let end = std::cmp::min(start + blocksize, nbytes);
        let block_len = end - start;
        let src_block = &src[start..end];

        bstarts[i] = current_dest_offset;

        let mut filtered_buf = if doshuffle != BLOSC_NOSHUFFLE as i32 {
            vec![0u8; block_len]
        } else {
            Vec::new()
        };

        let mut filtered_src = src_block;

        if doshuffle == BLOSC_SHUFFLE as i32 {
            filters::shuffle(typesize, block_len, src_block, &mut filtered_buf);
            filtered_src = &filtered_buf;
        } else if doshuffle == BLOSC_BITSHUFFLE as i32 {
            filters::bitshuffle(typesize, block_len, src_block, &mut filtered_buf)
                .map_err(|_| -1)?;
            filtered_src = &filtered_buf;
        }

        // C does not split the leftover (last partial) block
        let leftoverblock = i == nblocks - 1 && (nbytes % blocksize) != 0;
        let block_split =
            !leftoverblock && split_block(compressor, clevel, typesize, block_len, filter_flags, extended_header);
        let nstreams = if block_split { typesize } else { 1 };
        let neblock = block_len / nstreams;

        for j in 0..nstreams {
            let stream_offset = j * neblock;
            let stream_src = &filtered_src[stream_offset..stream_offset + neblock];

            if current_dest_offset + 4 > dest.len() {
                incompressible = true;
                break;
            }

            let stream_csize;

            match compressor {
                BLOSC_BLOSCLZ => {
                    stream_csize =
                        blosclz::compress(clevel, stream_src, &mut dest[current_dest_offset + 4..]);
                }
                BLOSC_LZ4 | BLOSC_LZ4HC => {
                    match lz4_flex::block::compress_into(
                        stream_src,
                        &mut dest[current_dest_offset + 4..],
                    ) {
                        Ok(size) => stream_csize = size,
                        Err(_) => stream_csize = 0,
                    }
                }
                BLOSC_SNAPPY => {
                    let mut encoder = snap::raw::Encoder::new();
                    match encoder.compress(stream_src, &mut dest[current_dest_offset + 4..]) {
                        Ok(size) => stream_csize = size,
                        Err(_) => stream_csize = 0,
                    }
                }
                BLOSC_ZLIB => {
                    let cursor = std::io::Cursor::new(&mut dest[current_dest_offset + 4..]);
                    let mut encoder = flate2::write::ZlibEncoder::new(
                        cursor,
                        flate2::Compression::new(clevel as u32),
                    );
                    if encoder.write_all(stream_src).is_ok() {
                        match encoder.finish() {
                            Ok(cursor) => {
                                stream_csize = cursor.position() as usize;
                            }
                            Err(_) => stream_csize = 0,
                        }
                    } else {
                        stream_csize = 0;
                    }
                }
                BLOSC_ZSTD => {
                    let cursor = std::io::Cursor::new(&mut dest[current_dest_offset + 4..]);
                    let mut encoder =
                        zstd::stream::write::Encoder::new(cursor, clevel).map_err(|_| -1)?;
                    if encoder.write_all(stream_src).is_ok() {
                        match encoder.finish() {
                            Ok(cursor) => {
                                stream_csize = cursor.position() as usize;
                            }
                            Err(_) => stream_csize = 0,
                        }
                    } else {
                        stream_csize = 0;
                    }
                }
                _ => return Err(-1),
            }

            if stream_csize == 0 || stream_csize >= neblock {
                incompressible = true;
                break;
            }

            dest[current_dest_offset..current_dest_offset + 4]
                .copy_from_slice(&(stream_csize as u32).to_le_bytes());
            current_dest_offset += 4 + stream_csize;
        }

        if incompressible {
            break;
        }
    }

    let compressed_size = current_dest_offset - data_offset;

    if incompressible || compressed_size >= nbytes {
        if nbytes > dest.len() - header_len {
            return Err(-1);
        }
        dest[header_len..header_len + nbytes].copy_from_slice(src);
        flags |= BLOSC_MEMCPYED;
        current_dest_offset = header_len + nbytes;
    } else {
        if nblocks > 0 {
            for i in 0..nblocks {
                let offset = header_len + i * 4;
                dest[offset..offset + 4].copy_from_slice(&(bstarts[i] as u32).to_le_bytes());
            }
        }
    }

    let cbytes = current_dest_offset;
    if extended_header {
        let header = create_header_blosc2(
            nbytes,
            blocksize,
            cbytes,
            typesize,
            flags,
            compressor,
            filters,
            filters_meta,
        );
        dest[0..BLOSC_EXTENDED_HEADER_LENGTH].copy_from_slice(&header);
    } else {
        let header = create_header_blosc1(nbytes, blocksize, cbytes, typesize, flags, compressor);
        dest[0..BLOSC_MIN_HEADER_LENGTH].copy_from_slice(&header);
    }

    Ok(cbytes)
}

/// Decompresses a Blosc/Blosc2 compressed buffer (including header) into `dest`.
///
/// Automatically detects the format version, codec, and filter pipeline from the
/// header. Supports both Blosc1 (16-byte header) and Blosc2 (32-byte extended header)
/// formats.
///
/// Returns the number of decompressed bytes written to `dest`.
pub fn decompress(src: &[u8], dest: &mut [u8]) -> Result<usize, Box<dyn std::error::Error>> {
    if src.len() < BLOSC_MIN_HEADER_LENGTH {
        return Err("Source buffer too small for header".into());
    }

    // Read version to determine header size
    let version = src[0];
    let header_len = if version == BLOSC2_VERSION_FORMAT_STABLE
        || version == BLOSC2_VERSION_FORMAT_BETA1
        || version == BLOSC2_VERSION_FORMAT_ALPHA
    {
        BLOSC_EXTENDED_HEADER_LENGTH
    } else {
        BLOSC_MIN_HEADER_LENGTH
    };

    if src.len() < header_len {
        return Err("Source buffer too small for extended header".into());
    }

    let (nbytes, cbytes, blocksize) = crate::api::blosc2_cbuffer_sizes(src);

    if dest.len() < nbytes {
        return Err("Destination buffer too small".into());
    }

    let flags = src[2];
    let is_blosc1 = header_len == BLOSC_MIN_HEADER_LENGTH;
    // Flags byte bits 5-7 store compformat, not compcode
    let compressor = if !is_blosc1 {
        // Extended header byte 22 has the actual compressor code
        src[22]
    } else {
        compformat_to_compcode((flags >> 5) & 0x7, is_blosc1)
    };
    let typesize = src[3] as usize;

    let mut doshuffle = (flags & BLOSC_DOSHUFFLE) != 0;
    let mut dobitshuffle = (flags & BLOSC_DOBITSHUFFLE) != 0;

    // Check for extended header marker (both flags set)
    if (flags & BLOSC_DOSHUFFLE) != 0 && (flags & BLOSC_DOBITSHUFFLE) != 0 {
        doshuffle = false;
        dobitshuffle = false;

        if !is_blosc1 {
            let filters = &src[16..22];
            for &f in filters {
                if f == BLOSC_SHUFFLE {
                    doshuffle = true;
                } else if f == BLOSC_BITSHUFFLE {
                    dobitshuffle = true;
                }
            }
        }
    }

    doshuffle = doshuffle && typesize > 1;
    dobitshuffle = dobitshuffle && blocksize >= typesize;

    if (flags & BLOSC_MEMCPYED) != 0 {
        // Copy from after header to dest
        dest[0..nbytes].copy_from_slice(&src[header_len..header_len + nbytes]);
        return Ok(nbytes);
    }

    // Calculate number of blocks
    let nblocks = if blocksize == 0 {
        0
    } else {
        (nbytes + blocksize - 1) / blocksize
    };

    // For extended headers with actual compression (not memcpy), read bstarts array
    let mut bstarts = Vec::new();

    if nblocks > 0 {
        // Always read bstarts array if nblocks > 0
        // Note: This assumes Blosc1 format also includes bstarts for nblocks=1.
        // c-blosc implementation suggests it does.
        if src.len() < header_len + nblocks * 4 {
            return Err("Buffer too small for bstarts array".into());
        }

        for i in 0..nblocks {
            let offset = header_len + i * 4;
            let bstart = u32::from_le_bytes([
                src[offset],
                src[offset + 1],
                src[offset + 2],
                src[offset + 3],
            ]) as usize;
            bstarts.push(bstart);
        }
    }

    // Determine split mode from header flags (bit 4 = dont_split)
    let dont_split = (flags & 0x10) != 0;

    // Decompress each block
    let mut dest_offset = 0;

    for i in 0..nblocks {
        let src_offset = bstarts[i];

        // Determine block size in compressed buffer
        let block_cbytes = if i + 1 < nblocks {
            bstarts[i + 1] - src_offset
        } else {
            cbytes - src_offset
        };

        // Determine uncompressed block size
        let leftoverblock = i == nblocks - 1 && nbytes % blocksize != 0;
        let block_nbytes = if leftoverblock {
            nbytes % blocksize // Last block may be smaller
        } else {
            blocksize
        };

        // Determine number of streams (matching C: dont_split or leftoverblock → 1 stream)
        let nstreams = if !dont_split && !leftoverblock {
            typesize
        } else {
            1
        };
        let neblock = block_nbytes / nstreams;

        if src_offset + block_cbytes > src.len() {
            return Err("Compressed data is truncated".into());
        }

        let content = &src[src_offset..src_offset + block_cbytes];

        let mut content_offset = 0;
        let mut block_dest_offset = 0;

        let use_temp = doshuffle || dobitshuffle;
        let mut temp_buf = if use_temp {
            vec![0u8; block_nbytes]
        } else {
            Vec::new()
        };
        {
            let target_slice = if use_temp {
                &mut temp_buf[..]
            } else {
                &mut dest[dest_offset..dest_offset + block_nbytes]
            };

            for _j in 0..nstreams {
                if content_offset + 4 > block_cbytes {
                    return Err("Block too small for chunk size".into());
                }
                // Read stream size as signed i32 (C uses sw32_ which returns signed)
                let stream_cbytes = i32::from_le_bytes([
                    content[content_offset],
                    content[content_offset + 1],
                    content[content_offset + 2],
                    content[content_offset + 3],
                ]);
                content_offset += 4;

                if stream_cbytes == 0 {
                    // A run of zeros
                    target_slice[block_dest_offset..block_dest_offset + neblock].fill(0);
                    block_dest_offset += neblock;
                } else if stream_cbytes < 0 {
                    // Run-length encoding: negative value encodes the byte value
                    if content_offset >= block_cbytes {
                        return Err("Not enough input for run-length token".into());
                    }
                    let token = content[content_offset];
                    content_offset += 1;

                    if token & 0x1 != 0 {
                        // A run of a non-zero byte value
                        let value = (-stream_cbytes) as u8;
                        target_slice[block_dest_offset..block_dest_offset + neblock].fill(value);
                    } else {
                        return Err("Invalid run-length token".into());
                    }
                    block_dest_offset += neblock;
                } else if stream_cbytes as usize == neblock {
                    // Incompressible: raw data stored directly
                    if content_offset + neblock > block_cbytes {
                        return Err("Chunk size exceeds block size".into());
                    }
                    target_slice[block_dest_offset..block_dest_offset + neblock]
                        .copy_from_slice(&content[content_offset..content_offset + neblock]);
                    content_offset += neblock;
                    block_dest_offset += neblock;
                } else {
                    // Compressed data
                    let sc = stream_cbytes as usize;
                    if content_offset + sc > block_cbytes {
                        return Err("Chunk size exceeds block size".into());
                    }

                    let chunk_content = &content[content_offset..content_offset + sc];
                    content_offset += sc;

                    let dest_slice =
                        &mut target_slice[block_dest_offset..block_dest_offset + neblock];

                    let chunk_decompressed_size = match compressor {
                        BLOSC_BLOSCLZ => blosclz::decompress(chunk_content, dest_slice),
                        BLOSC_LZ4 | BLOSC_LZ4HC => {
                            lz4_flex::decompress_into(chunk_content, dest_slice)
                                .map_err(|e| format!("LZ4 error: {}", e))?
                        }
                        BLOSC_SNAPPY => {
                            let mut decoder = snap::raw::Decoder::new();
                            decoder
                                .decompress(chunk_content, dest_slice)
                                .map_err(|e| format!("Snappy error: {}", e))?
                        }
                        BLOSC_ZLIB => {
                            let mut decoder = flate2::read::ZlibDecoder::new(chunk_content);
                            let mut writer = std::io::Cursor::new(dest_slice);
                            std::io::copy(&mut decoder, &mut writer)
                                .map_err(|e| format!("Zlib error: {}", e))?
                                as usize
                        }
                        BLOSC_ZSTD => zstd::bulk::decompress_to_buffer(chunk_content, dest_slice)
                            .map_err(|e| format!("Zstd error: {}", e))?,
                        _ => return Err(format!("Unsupported compressor: {}", compressor).into()),
                    };

                    block_dest_offset += chunk_decompressed_size;
                }
            }

            if block_dest_offset != block_nbytes {
                return Err(format!(
                    "Block {} decompression size mismatch: expected {}, got {}",
                    i, block_nbytes, block_dest_offset
                )
                .into());
            }
        }

        if use_temp {
            if doshuffle {
                filters::unshuffle(
                    typesize,
                    block_nbytes,
                    &temp_buf,
                    &mut dest[dest_offset..dest_offset + block_nbytes],
                );
            } else {
                filters::bitunshuffle(
                    typesize,
                    block_nbytes,
                    &temp_buf,
                    &mut dest[dest_offset..dest_offset + block_nbytes],
                )
                .map_err(|e| format!("Bitunshuffle error: {}", e))?;
            }
        }

        dest_offset += block_dest_offset;
    }

    Ok(nbytes)
}

/// Partially decompresses a Blosc buffer, extracting `nitems` elements starting
/// at element index `start`.
///
/// This avoids decompressing the entire buffer when only a subset of elements is
/// needed. Returns the number of bytes written to `dest`.
pub fn getitem(src: &[u8], start: usize, nitems: usize, dest: &mut [u8]) -> Result<usize, i32> {
    if src.len() < BLOSC_MIN_HEADER_LENGTH {
        return Err(-1);
    }

    // Check version to determine header size
    let version = src[0];
    let header_len = if version == BLOSC2_VERSION_FORMAT_STABLE
        || version == BLOSC2_VERSION_FORMAT_BETA1
        || version == BLOSC2_VERSION_FORMAT_ALPHA
    {
        BLOSC_EXTENDED_HEADER_LENGTH
    } else {
        BLOSC_MIN_HEADER_LENGTH
    };

    let flags = src[2];
    let is_blosc1 = header_len == BLOSC_MIN_HEADER_LENGTH;
    // Flags byte bits 5-7 store compformat, not compcode
    let compressor = if !is_blosc1 {
        src[22]
    } else {
        compformat_to_compcode((flags >> 5) & 0x7, is_blosc1)
    };
    let typesize = src[3] as usize;
    let nbytes = u32::from_le_bytes(src[4..8].try_into().unwrap()) as usize;
    let blocksize = u32::from_le_bytes(src[8..12].try_into().unwrap()) as usize;
    let cbytes = u32::from_le_bytes(src[12..16].try_into().unwrap()) as usize;

    if src.len() < cbytes {
        return Err(-1);
    }

    let start_byte = start * typesize;
    let end_byte = (start + nitems) * typesize;
    if end_byte > nbytes {
        return Err(-1);
    }
    if dest.len() < end_byte - start_byte {
        return Err(-1);
    }

    if (flags & BLOSC_MEMCPYED) != 0 {
        dest.copy_from_slice(&src[header_len + start_byte..header_len + end_byte]);
        return Ok(end_byte - start_byte);
    }

    // Determine shuffle mode (same logic as decompress)
    let mut doshuffle = (flags & BLOSC_DOSHUFFLE) != 0;
    let mut dobitshuffle = (flags & BLOSC_DOBITSHUFFLE) != 0;
    if (flags & BLOSC_DOSHUFFLE) != 0 && (flags & BLOSC_DOBITSHUFFLE) != 0 {
        doshuffle = false;
        dobitshuffle = false;
        if !is_blosc1 {
            for &f in &src[16..22] {
                if f == BLOSC_SHUFFLE {
                    doshuffle = true;
                } else if f == BLOSC_BITSHUFFLE {
                    dobitshuffle = true;
                }
            }
        }
    }
    doshuffle = doshuffle && typesize > 1;
    dobitshuffle = dobitshuffle && blocksize >= typesize;

    let dont_split = (flags & 0x10) != 0;

    let nblocks = if nbytes == 0 {
        0
    } else {
        (nbytes + blocksize - 1) / blocksize
    };

    let start_block = start_byte / blocksize;
    let end_block = (end_byte - 1) / blocksize;

    // Read bstarts array
    let mut bstarts = Vec::new();
    if nblocks > 0 {
        if src.len() < header_len + nblocks * 4 {
            return Err(-1);
        }
        for i in 0..nblocks {
            let off = header_len + i * 4;
            let bstart = u32::from_le_bytes(src[off..off + 4].try_into().unwrap()) as usize;
            bstarts.push(bstart);
        }
    }

    let mut dest_offset = 0;

    for i in start_block..=end_block {
        let block_start_off = bstarts[i];
        let block_end_off = if i + 1 < nblocks {
            bstarts[i + 1]
        } else {
            cbytes
        };
        let block_cbytes = block_end_off - block_start_off;

        let leftoverblock = i == nblocks - 1 && nbytes % blocksize != 0;
        let current_block_size = if leftoverblock {
            nbytes % blocksize
        } else {
            blocksize
        };

        let nstreams = if !dont_split && !leftoverblock {
            typesize
        } else {
            1
        };
        let neblock = current_block_size / nstreams;

        // Decompress the whole block using stream-aware logic
        let content = &src[block_start_off..block_start_off + block_cbytes];
        let use_temp = doshuffle || dobitshuffle;
        let mut raw_buf = vec![0u8; current_block_size];
        {
            let target = if use_temp {
                &mut raw_buf[..]
            } else {
                &mut raw_buf[..]
            };

            let mut coff = 0usize;
            let mut doff = 0usize;
            for _j in 0..nstreams {
                if coff + 4 > block_cbytes {
                    return Err(-1);
                }
                let stream_cb = i32::from_le_bytes(content[coff..coff + 4].try_into().unwrap());
                coff += 4;

                if stream_cb == 0 {
                    target[doff..doff + neblock].fill(0);
                    doff += neblock;
                } else if stream_cb < 0 {
                    if coff >= block_cbytes {
                        return Err(-1);
                    }
                    let token = content[coff];
                    coff += 1;
                    if token & 0x1 != 0 {
                        let value = (-stream_cb) as u8;
                        target[doff..doff + neblock].fill(value);
                    } else {
                        return Err(-1);
                    }
                    doff += neblock;
                } else if stream_cb as usize == neblock {
                    if coff + neblock > block_cbytes {
                        return Err(-1);
                    }
                    target[doff..doff + neblock].copy_from_slice(&content[coff..coff + neblock]);
                    coff += neblock;
                    doff += neblock;
                } else {
                    let sc = stream_cb as usize;
                    if coff + sc > block_cbytes {
                        return Err(-1);
                    }
                    let chunk = &content[coff..coff + sc];
                    coff += sc;
                    let ds = &mut target[doff..doff + neblock];
                    let n = match compressor {
                        BLOSC_BLOSCLZ => blosclz::decompress(chunk, ds),
                        BLOSC_LZ4 | BLOSC_LZ4HC => {
                            lz4_flex::decompress_into(chunk, ds).map_err(|_| -1)?
                        }
                        BLOSC_SNAPPY => {
                            let mut dec = snap::raw::Decoder::new();
                            dec.decompress(chunk, ds).map_err(|_| -1)?
                        }
                        BLOSC_ZLIB => {
                            let mut dec = flate2::read::ZlibDecoder::new(chunk);
                            let mut w = std::io::Cursor::new(ds);
                            std::io::copy(&mut dec, &mut w).map_err(|_| -1)? as usize
                        }
                        BLOSC_ZSTD => {
                            zstd::bulk::decompress_to_buffer(chunk, ds).map_err(|_| -1)?
                        }
                        _ => return Err(-1),
                    };
                    doff += n;
                }
            }
            if doff != current_block_size {
                return Err(-1);
            }
        }

        // Apply unshuffle if needed
        let block_buf = if use_temp {
            let mut buf = vec![0u8; current_block_size];
            if doshuffle {
                filters::unshuffle(typesize, current_block_size, &raw_buf, &mut buf);
            } else if dobitshuffle {
                filters::bitunshuffle(typesize, current_block_size, &raw_buf, &mut buf)
                    .map_err(|_| -1)?;
            }
            buf
        } else {
            raw_buf
        };

        // Copy relevant part
        let block_start = i * blocksize;
        let block_end = block_start + current_block_size;

        let copy_start = std::cmp::max(start_byte, block_start);
        let copy_end = std::cmp::min(end_byte, block_end);

        let local_start = copy_start - block_start;
        let local_end = copy_end - block_start;
        let len = local_end - local_start;

        dest[dest_offset..dest_offset + len].copy_from_slice(&block_buf[local_start..local_end]);
        dest_offset += len;
    }

    Ok(dest_offset)
}
