# My current understanding of the BLOSC algorithm

## Compression

The blosc2 compression pipeline works as follows:

1. **Block sizing** (`compute_blocksize` / C `blosc_stune_next_blocksize`):
   - Determines blocksize based on clevel, typesize, nbytes, and compressor.
   - Base is L1 cache size (32KB), scaled up by clevel.
   - HCR codecs (LZ4HC, ZLIB, ZSTD) get 2x blocksize.
   - When splitting is active (byte-shuffle + compatible codec), a separate blocksize
     table is used (32KB-512KB * typesize, capped at 4MB).

2. **Header creation**: 16-byte (blosc1) or 32-byte (blosc2 extended) header.
   - Byte 0: version format
   - Byte 1: compressor version
   - Byte 2: flags (shuffle bits, memcpy bit, codec in bits 5-7)
   - Byte 3: typesize
   - Bytes 4-7: nbytes (uncompressed size, LE u32)
   - Bytes 8-11: blocksize (LE u32)
   - Bytes 12-15: cbytes (compressed size, LE u32)
   - Extended header bytes 16-21: filter codes
   - Byte 22: compressor code
   - Bytes 24-29: filter metadata

3. **Split decision** (`split_block` / C `stune.c:split_block`):
   - Only splits for byte-shuffle (BLOSC_DOSHUFFLE), NOT bitshuffle.
   - Only for BLOSCLZ, LZ4, or ZSTD (clevel <= 5).
   - typesize must be <= 16 (MAX_STREAMS).
   - blocksize/typesize must be >= BLOSC_MIN_BUFFERSIZE (32).

4. **Per-block processing**:
   - Apply filter (shuffle or bitshuffle).
   - If splitting: compress each typesize-stream separately.
   - If not splitting: compress entire block as one stream.
   - Each stream is preceded by a 4-byte LE u32 compressed size.

5. **Incompressible fallback**: If any stream fails to compress or total compressed
   exceeds original, fall back to memcpy (BLOSC_MEMCPYED flag).

## Decompression

1. Parse header to get nbytes, cbytes, blocksize, flags, compressor, typesize.
2. Detect extended header (both DOSHUFFLE and DOBITSHUFFLE flags set = extended marker).
3. If MEMCPYED flag: direct copy from after header.
4. Otherwise: read bstarts array, decompress each block's streams, apply inverse filter.

## BloscLZ Codec

Reference: `c-blosc2/blosc/blosclz.c`

BloscLZ is based on FastLZ. Key implementation details:

### Compression (`blosclz_compress`, line ~430)
- Uses ipshift=4, minlen=4 (constant for all clevels).
- Hash table size: `1 << hashlog` where hashlog depends on clevel
  (0 for clevel 0, HASH_LOG-2 for 1, HASH_LOG-1 for 2, HASH_LOG for 3+).
- **Entropy probing** (line ~455): estimates compression ratio by running a simulated
  compression on a portion of the buffer. Bails early if ratio is too low.
  Probe length: length/8 (clevel<2), length/4 (clevel<4), length/2 (clevel<7), full (clevel>=7).
  Threshold table: [0, 2, 1.5, 1.2, 1.2, 1.2, 1.2, 1.15, 1.1, 1.0].
- Hash table is re-initialized to 0 after entropy probing (line ~485).
- Match detection: first 4 bytes checked, then `get_run_or_match` extends.
- `get_match` returns one-past-end (C post-increment semantics) -> encoded len = ip - anchor.
- `get_run` (for repeated bytes, distance==0 after decrement) does NOT post-increment.
- Encoded match length is biased: decompressor adds 3 to recover actual length.
- After encoding, hash is updated at match boundary position, then ip advances by 2.
- At clevel 9, a second hash entry is stored (shifted by 1 byte).

### Decompression (`blosclz_decompress`, line ~550)
- ctrl byte: if >= 32 -> match, else -> literal copy.
- Match: len = (ctrl >> 5) - 1, with extension for len==6. Then len += 3.
- Far distance (code==255, ofs==31*256): 16-bit distance encoding.
- Optimized copy for runs (distance==1) using memset.

### Critical C implementation detail: `get_match` post-increment
The C `get_match` function (line ~149) uses `while (*ref++ == *ip++) {}`.
The post-increment means ip advances one extra byte on mismatch but NOT when
hitting ip_bound. In the 8-byte fast path (non-STRICT_ALIGN), the inner byte-by-byte
fallback has no ip_bound check, so it always post-increments on mismatch.
