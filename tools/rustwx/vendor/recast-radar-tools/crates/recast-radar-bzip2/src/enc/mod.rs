//! The compressor: [`Encoder`] and [`Level`].

mod bits;
mod bwt;
mod huffman;
mod mtf;
#[cfg(feature = "rayon")]
mod par;
mod rle1;
mod sais;

#[cfg(feature = "rayon")]
pub use par::{EncoderPool, encode_many};

use crate::crc;
use bits::BitWriter;

const BLOCK_MAGIC: u64 = 0x3141_5926_5359;
const EOS_MAGIC: u64 = 0x1772_4538_5090;

/// bzip2 block size, `1` to `9`: blocks of up to `100_000 * level` bytes
/// after the initial run-length stage.
///
/// The level is written in the stream header (`BZh1`..`BZh9`). Larger
/// blocks usually compress better and need more memory to compress and to
/// decompress. NEXRAD Level II LDM records use level 9.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Level(u8);

impl Level {
    /// Level 1: 100,000-byte blocks.
    pub const FASTEST: Level = Level(1);
    /// Level 9: 900,000-byte blocks, the `bzip2` command's default.
    pub const BEST: Level = Level(9);

    /// The level `level`, or `None` unless `1 <= level <= 9`.
    #[must_use]
    pub const fn new(level: u32) -> Option<Level> {
        if level >= 1 && level <= 9 {
            Some(Level(level as u8))
        } else {
            None
        }
    }

    /// The level as a number, `1` to `9`.
    pub const fn get(self) -> u32 {
        self.0 as u32
    }

    /// Largest number of bytes a block can hold (`100_000 * level`).
    const fn block_capacity(self) -> usize {
        100_000 * self.0 as usize
    }
}

impl Default for Level {
    /// [`Level::BEST`].
    fn default() -> Self {
        Level::BEST
    }
}

/// A reusable bzip2 compressor.
///
/// [`Encoder::encode_into`] writes one complete bzip2 stream per call. The
/// stream is the one libbzip2 1.0.8 writes for the same input and block
/// size with `BZ2_bzBuffToBuffCompress` (any work factor), byte for byte,
/// except for one field in a block that is an exact repetition of a shorter
/// string (see the crate docs, *Compressing*).
///
/// The work buffers (about 22 bytes per byte of block capacity, so about
/// 2.5 MB at level 1 and 20 MB at level 9) are allocated on the first call
/// and reused, so keep one encoder per thread and reuse it; for many inputs
/// in parallel, see `EncoderPool` (feature `rayon`).
pub struct Encoder {
    level: Level,
    ws: Option<Box<Workspace>>,
    /// Test hook: bit offsets, from the start of the last stream written,
    /// of the `origPtr` fields of its periodic blocks.
    periodic: Vec<u64>,
}

struct Workspace {
    /// A pad byte, the RLE1 bytes of the block, then a second copy of them
    /// for the least-rotation search (see `bwt.rs`).
    block: Vec<u8>,
    /// Suffix array work space (`cap + 1`: the last entry is a dummy slot).
    sa: Vec<i32>,
    pool: Vec<i32>,
    bpool: Vec<u64>,
    /// BWT last column (with slack for word reads past its end).
    last: Vec<u8>,
    mtfv: Vec<u16>,
    tables: Box<huffman::Tables>,
}

impl Workspace {
    fn new(level: Level) -> Box<Workspace> {
        let cap = level.block_capacity();
        Box::new(Workspace {
            block: vec![0; 2 * cap + 16 + 1],
            sa: vec![0; cap + 1],
            pool: vec![0; sais::pool_len(cap)],
            bpool: vec![0; sais::bitmap_pool_len(cap)],
            last: vec![0; cap + 16],
            mtfv: vec![0; cap + 1],
            tables: huffman::Tables::new(),
        })
    }
}

impl core::fmt::Debug for Encoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Encoder")
            .field("level", &self.level)
            .finish_non_exhaustive()
    }
}

impl Encoder {
    /// An encoder for `level`. The work buffers are allocated on the first
    /// call to [`Encoder::encode_into`].
    pub fn new(level: Level) -> Encoder {
        Encoder {
            level,
            ws: None,
            periodic: Vec::new(),
        }
    }

    /// The block size this encoder writes.
    pub fn level(&self) -> Level {
        self.level
    }

    /// Compress `input` into one complete bzip2 stream (header, blocks,
    /// end-of-stream marker and combined CRC, padded to a byte boundary)
    /// appended to `out`. An empty input gives the 14-byte empty stream.
    pub fn encode_into(&mut self, input: &[u8], out: &mut Vec<u8>) {
        let level = self.level;
        let ws = self.ws.get_or_insert_with(|| Workspace::new(level));
        // libbzip2's nblockMAX.
        let nblock_max = level.block_capacity() - 19;
        out.reserve(input.len() / 4 + 64);
        let mut bw = BitWriter::new(out);
        bw.put(24, u32::from_be_bytes([0, b'B', b'Z', b'h']));
        bw.put(8, u32::from(b'0') + level.get());
        let mut combined = 0u32;
        let mut pos = 0usize;
        let mut in_use = [false; 256];
        let mut counts = [0u32; 256];
        self.periodic.clear();
        while pos < input.len() {
            let (nblock, end) =
                rle1::fill_block(input, pos, nblock_max, &mut ws.block[1..], &mut in_use);
            counts.fill(0);
            for &x in &ws.block[1..=nblock] {
                counts[x as usize] += 1;
            }
            let block_crc = !crc::update(!0, &input[pos..end]);
            combined = combined.rotate_left(1) ^ block_crc;
            encode_block(
                ws,
                nblock,
                &counts,
                block_crc,
                &in_use,
                &mut bw,
                &mut self.periodic,
            );
            pos = end;
        }
        bw.put48(EOS_MAGIC);
        bw.put(32, combined);
        bw.finish();
    }

    /// Test hook, not part of the supported API: the bit offsets, counted
    /// from the start of the stream written by the last
    /// [`Encoder::encode_into`] call, of the 24-bit `origPtr` field of every
    /// block that is an exact repetition of a shorter string. Only these
    /// fields can differ from libbzip2's output.
    #[doc(hidden)]
    pub fn __periodic_orig_ptr_bits(&self) -> &[u64] {
        &self.periodic
    }
}

fn encode_block(
    ws: &mut Workspace,
    nblock: usize,
    counts: &[u32; 256],
    block_crc: u32,
    in_use: &[bool; 256],
    bw: &mut BitWriter<'_>,
    periodic: &mut Vec<u64>,
) {
    let t = bwt::bwt(
        &mut ws.block,
        nblock,
        counts,
        &mut ws.sa,
        &mut ws.last,
        &mut ws.pool,
        &mut ws.bpool,
    );
    bw.put48(BLOCK_MAGIC);
    bw.put(32, block_crc);
    bw.put(1, 0);
    if t.periodic {
        periodic.push(bw.bit_len());
    }
    bw.put(24, t.orig_ptr);
    let mut freq = [0u32; 258];
    let (n_mtf, n_in_use) = mtf::encode(&ws.last, nblock, in_use, &mut ws.mtfv, &mut freq);
    huffman::send(
        &mut ws.tables,
        &ws.mtfv[..n_mtf],
        &freq,
        in_use,
        n_in_use,
        bw,
    );
}
