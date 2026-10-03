//! A whole-file bzip2 stream decoded one block at a time.
//!
//! `recast-radar-bzip2` decodes whole streams only. A bzip2 stream is a
//! sequence of independent blocks, each starting with the 48-bit magic
//! `0x314159265359` and its CRC-32 at an arbitrary bit offset, and ending
//! where the next block's magic or the end-of-stream magic `0x177245385090`
//! begins. [`decode_prefix`] and [`Bzip2RecordBytes`] find those
//! boundaries, copy each block's bits into a one-block stream of its own
//! (header `BZh<level>`, the block, the end-of-stream magic and, as combined
//! CRC, the block's CRC) and decode blocks in order: [`decode_prefix`] until
//! the requested length is covered (the metadata record), and
//! [`Bzip2RecordBytes`] as the record parser asks for bytes (the volume
//! decode, which then never holds the whole expanded stream).
//!
//! Every block is decoded with its CRC checked, so a boundary found by
//! mistake (the magic's bit pattern inside compressed data) cannot produce
//! wrong bytes: its block fails to decode, and the stream is decoded whole
//! as before ([`decode_prefix`] returns `None` to its caller;
//! [`Bzip2RecordBytes`] switches to the whole decode itself).

use crate::{
    MAX_DECODED_RADAR_BYTES, NexradError, RecordBytes, Result, decompress_bzip2_stream_into,
};

const BLOCK_MAGIC: u64 = 0x3141_5926_5359;
const END_MAGIC: u64 = 0x1772_4538_5090;
const MAGIC_BITS: usize = 48;
/// A block header: magic, CRC-32, randomised bit, 24-bit origin pointer.
const BLOCK_HEADER_BITS: usize = MAGIC_BITS + 32 + 1 + 24;

/// The first `prefix_len` (or more) decoded bytes of the bzip2 stream at the
/// start of `input`, and whether they are the whole stream. `None` when the
/// stream cannot be decoded block by block; the caller then decodes it
/// whole, which reports any error. At most [`MAX_DECODED_RADAR_BYTES`] are
/// decoded.
pub(crate) fn decode_prefix(
    input: &[u8],
    prefix_len: usize,
    context: &'static str,
) -> Result<Option<(Vec<u8>, bool)>> {
    let Some(&level) = input.get(3) else {
        return Ok(None);
    };
    if !input.starts_with(b"BZh") || !(b'1'..=b'9').contains(&level) {
        return Ok(None);
    }
    let finder = MagicFinder::new();
    let mut output = Vec::new();
    let mut block_start = 32;
    // The stream CRC as the decoder combines it from the block CRCs.
    let mut combined: u32 = 0;
    loop {
        match bits_at(input, block_start, MAGIC_BITS) {
            Some(BLOCK_MAGIC) => {}
            // The whole stream is decoded: check its stored CRC as the
            // decoder does.
            Some(END_MAGIC) => {
                let stored = bits_at(input, block_start + MAGIC_BITS, 32);
                return Ok((stored == Some(u64::from(combined))).then_some((output, true)));
            }
            _ => return Ok(None),
        }
        let Some((next, _)) = finder.next_magic(input, block_start + BLOCK_HEADER_BITS) else {
            return Ok(None);
        };
        let Some(crc) = bits_at(input, block_start + MAGIC_BITS, 32) else {
            return Ok(None);
        };
        let stream = one_block_stream(input, level, block_start, next, crc);
        let remaining = MAX_DECODED_RADAR_BYTES.saturating_sub(output.len());
        if decompress_bzip2_stream_into(&stream, &mut output, remaining, context).is_err() {
            return Ok(None);
        }
        combined = combined.rotate_left(1) ^ crc as u32;
        if output.len() >= prefix_len && bits_at(input, next, MAGIC_BITS) != Some(END_MAGIC) {
            return Ok(Some((output, false)));
        }
        block_start = next;
    }
}

/// A whole-file bzip2 stream decoded for the record parser one block at a
/// time: the bytes the parser has not released and the last block decoded,
/// in one window.
///
/// The result is exactly that of decoding the stream whole
/// ([`decompress_bzip2_stream_into`] with the same limit and context): the
/// same bytes, the stored combined CRC checked at the end-of-stream magic,
/// bytes after it ignored, and on any failure of the block-wise decode (a
/// block that does not decode, a boundary not found, a combined CRC that
/// does not match, the output limit) the stream is decoded whole, which
/// either reports the one-shot decode's error or, after a boundary found by
/// mistake, supplies the rest of the bytes. Blocks already decoded were
/// CRC-checked, so they are the start of the whole decode.
/// [`Self::finish_after`] decodes whatever the parser did not read, so a
/// volume decodes exactly when the whole decode would have succeeded.
pub(crate) struct Bzip2RecordBytes<'a> {
    input: &'a [u8],
    /// The stream header's level digit, for each one-block stream.
    level: u8,
    finder: MagicFinder,
    /// Bit offset of the next block's magic (or of the end-of-stream magic).
    next_block: usize,
    /// The stream CRC combined from the decoded blocks' CRCs.
    combined: u32,
    /// `window` holds decoded bytes `base..base + window.len()`.
    window: Vec<u8>,
    base: usize,
    /// Offset before which the parser reads nothing again.
    released: usize,
    /// No more output: the stream ended, was decoded whole, or failed.
    done: bool,
    /// The whole decode's error, returned again by later calls.
    error: Option<String>,
    limit: usize,
    context: &'static str,
}

impl<'a> Bzip2RecordBytes<'a> {
    pub(crate) fn new(input: &'a [u8], limit: usize, context: &'static str) -> Self {
        Self {
            input,
            level: input.get(3).copied().unwrap_or(0),
            finder: MagicFinder::new(),
            next_block: 32,
            combined: 0,
            window: Vec::new(),
            base: 0,
            released: 0,
            done: false,
            error: None,
            limit,
            context,
        }
    }

    /// Decoded bytes so far.
    fn produced(&self) -> usize {
        self.base + self.window.len()
    }

    /// Decode the next block into the window, or finish at the end of the
    /// stream; on any failure decode the stream whole instead.
    fn decode_next_block(&mut self) {
        if self.done {
            return;
        }
        if !self.input.starts_with(b"BZh") || !(b'1'..=b'9').contains(&self.level) {
            return self.decode_whole();
        }
        let start = self.next_block;
        match bits_at(self.input, start, MAGIC_BITS) {
            Some(BLOCK_MAGIC) => {}
            Some(END_MAGIC) => {
                let stored = bits_at(self.input, start + MAGIC_BITS, 32);
                if stored == Some(u64::from(self.combined)) {
                    self.done = true;
                    return;
                }
                return self.decode_whole();
            }
            _ => return self.decode_whole(),
        }
        let (Some((next, _)), Some(crc)) = (
            self.finder
                .next_magic(self.input, start + BLOCK_HEADER_BITS),
            bits_at(self.input, start + MAGIC_BITS, 32),
        ) else {
            return self.decode_whole();
        };
        // Drop the released prefix before the window grows: what remains is
        // the part of one message the parser has not finished.
        let release = self
            .released
            .saturating_sub(self.base)
            .min(self.window.len());
        if release > 0 {
            self.window.drain(..release);
            self.base += release;
        }
        let stream = one_block_stream(self.input, self.level, start, next, crc);
        let remaining = self.limit.saturating_sub(self.produced());
        // On error the window is back at its length before the call.
        if decompress_bzip2_stream_into(&stream, &mut self.window, remaining, self.context).is_err()
        {
            return self.decode_whole();
        }
        self.combined = self.combined.rotate_left(1) ^ crc as u32;
        self.next_block = next;
    }

    /// Decode the stream whole, as the one-shot path does: its error, or the
    /// bytes after the ones already decoded.
    fn decode_whole(&mut self) {
        self.done = true;
        let mut whole = Vec::new();
        match decompress_bzip2_stream_into(self.input, &mut whole, self.limit, self.context) {
            Ok(()) => {
                // Every block decoded so far passed its CRC, so the window is
                // `whole[base..]` up to its length: the window becomes
                // `whole[base..]`.
                let produced = self.produced();
                if produced <= whole.len() {
                    whole.drain(..self.base);
                    self.window = whole;
                } else {
                    self.error = Some(format!(
                        "{}: block-wise decode produced {produced} bytes, whole decode {}",
                        self.context,
                        whole.len()
                    ));
                }
            }
            Err(NexradError::Compression(message)) => self.error = Some(message),
            Err(other) => self.error = Some(other.to_string()),
        }
    }

    /// Decode the rest of the stream, discarding it, and return `parsed`
    /// unless the stream fails to decode, whose error then wins (the whole
    /// decode reported it before any parsing).
    pub(crate) fn finish_after<T>(mut self, parsed: Result<T>) -> Result<T> {
        while !self.done {
            self.base += self.window.len();
            self.window.clear();
            self.released = self.base;
            self.decode_next_block();
        }
        match self.error {
            Some(error) => Err(NexradError::Compression(error)),
            None => parsed,
        }
    }
}

impl RecordBytes for Bzip2RecordBytes<'_> {
    fn extend_to(&mut self, end: usize) -> Result<usize> {
        while self.produced() < end && !self.done {
            self.decode_next_block();
        }
        if self.produced() < end
            && let Some(error) = &self.error
        {
            return Err(NexradError::Compression(error.clone()));
        }
        Ok(end.min(self.produced()))
    }

    fn get(&self, start: usize, end: usize) -> Result<&[u8]> {
        crate::window_range(&self.window, self.base, start, end)
    }

    fn release_before(&mut self, offset: usize) {
        self.released = self.released.max(offset);
    }
}

/// `count` (at most 57) bits of `input` starting at bit `bit`, most
/// significant first; `None` past the end.
fn bits_at(input: &[u8], bit: usize, count: usize) -> Option<u64> {
    debug_assert!((1..=57).contains(&count));
    let end = bit.checked_add(count)?;
    if end > input.len().checked_mul(8)? {
        return None;
    }
    let first = bit / 8;
    let mut word = [0u8; 8];
    let available = input.len() - first;
    word[..available.min(8)].copy_from_slice(&input[first..first + available.min(8)]);
    let value = u64::from_be_bytes(word) << (bit % 8);
    Some(value >> (64 - count))
}

/// A one-block bzip2 stream: `BZh<level>`, bits `start..end` of `input`
/// (the block), the end-of-stream magic and `crc` as the combined CRC.
fn one_block_stream(input: &[u8], level: u8, start: usize, end: usize, crc: u64) -> Vec<u8> {
    let mut writer = BitWriter::with_capacity((end - start) / 8 + 16);
    writer.bytes.extend_from_slice(&[b'B', b'Z', b'h', level]);
    // The block's whole bytes realigned in one pass (byte `k` is bits
    // `start + 8k..start + 8k + 8`), then the few bits left over.
    let whole = (end - start) / 8;
    let first = start / 8;
    let shift = (start % 8) as u32;
    let aligned = if shift == 0 {
        input
            .get(first..first + whole)
            .map(|bytes| writer.bytes.extend_from_slice(bytes))
    } else {
        // Byte `first + whole` holds the last whole byte's low bits: it is
        // inside the input because `end` is.
        input.get(first..=first + whole).map(|bytes| {
            writer.bytes.extend(
                bytes
                    .windows(2)
                    .map(|pair| (pair[0] << shift) | (pair[1] >> (8 - shift))),
            )
        })
    };
    let mut bit = if aligned.is_some() {
        start + whole * 8
    } else {
        start
    };
    while bit < end {
        let count = (end - bit).min(56);
        // In range: `end` is at most the input's bit length.
        writer.write(bits_at(input, bit, count).unwrap_or(0), count);
        bit += count;
    }
    writer.write(END_MAGIC, MAGIC_BITS);
    writer.write(crc, 32);
    writer.finish()
}

/// Most-significant-first bit writer.
struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    bits: usize,
}

impl BitWriter {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
            acc: 0,
            bits: 0,
        }
    }

    /// Append the low `count` (at most 56) bits of `value`.
    fn write(&mut self, value: u64, count: usize) {
        debug_assert!(count <= 56);
        let mask = (1u64 << count) - 1;
        self.acc = (self.acc << count) | (value & mask);
        self.bits += count;
        while self.bits >= 8 {
            self.bits -= 8;
            self.bytes.push((self.acc >> self.bits) as u8);
        }
        self.acc &= (1u64 << self.bits) - 1;
    }

    /// The bytes, the last one padded with zero bits.
    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.bytes.push((self.acc << (8 - self.bits)) as u8);
        }
        self.bytes
    }
}

/// The magics [`MagicFinder`] looks for.
const MAGICS: [u64; 2] = [BLOCK_MAGIC, END_MAGIC];

/// Finds the next block or end-of-stream magic at any bit offset.
///
/// A magic starting at bit `8 * j + s` fills bytes `j + 1..=j + 5` whole,
/// each with a value that depends only on the magic, `s` and the byte's
/// place `k` in it. One of `j + 1..=j + 4` is a multiple of four, so only
/// every fourth byte is looked at: a byte and the one after it are looked up
/// in two tables of the (magic, `s`, `k`) combinations they could be, and
/// only a combination both admit is checked in full. The scan costs about
/// one table pair per four input bytes.
struct MagicFinder {
    /// For each byte value, the combinations in which a magic's byte `k`
    /// has that value, one bit each: bit `32 * m + 8 * (k - 1) + s` for
    /// magic `m` of [`MAGICS`] starting at bit shift `s`.
    this: [u64; 256],
    /// The same for the byte after it (the magic's byte `k + 1`).
    next: [u64; 256],
}

impl MagicFinder {
    fn new() -> Self {
        let mut this = [0u64; 256];
        let mut next = [0u64; 256];
        for (m, magic) in MAGICS.iter().enumerate() {
            for k in 1..=4usize {
                for s in 0..8usize {
                    let combination = 1u64 << (m * 32 + (k - 1) * 8 + s);
                    // Byte `j + k` holds magic bits `8k - s..8k - s + 8`.
                    let byte = |k: usize| ((magic >> (40 - 8 * k + s)) & 0xff) as usize;
                    this[byte(k)] |= combination;
                    next[byte(k + 1)] |= combination;
                }
            }
        }
        Self { this, next }
    }

    /// The first bit offset at or after `from` where either magic starts,
    /// with the magic found.
    fn next_magic(&self, input: &[u8], from: usize) -> Option<(usize, u64)> {
        // A magic at or after `from` has a byte `j + k` (k in 1..=4) at a
        // multiple of four at or after `from / 8 + 1`.
        let mut index = (from / 8 + 1).next_multiple_of(4);
        while index + 1 < input.len() {
            let mut candidates =
                self.this[usize::from(input[index])] & self.next[usize::from(input[index + 1])];
            let mut found: Option<(usize, u64)> = None;
            while candidates != 0 {
                let combination = candidates.trailing_zeros() as usize;
                candidates &= candidates - 1;
                let (m, k, s) = (
                    combination / 32,
                    (combination % 32) / 8 + 1,
                    combination % 8,
                );
                let Some(bit) = (index * 8 + s).checked_sub(8 * k) else {
                    continue;
                };
                if bit < from || found.is_some_and(|(earliest, _)| earliest <= bit) {
                    continue;
                }
                if bits_at(input, bit, MAGIC_BITS) == Some(MAGICS[m]) {
                    found = Some((bit, MAGICS[m]));
                }
            }
            // Magics found at a later index start later than any found here.
            if found.is_some() {
                return found;
            }
            index += 4;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn compress(payload: &[u8], level: u32) -> Vec<u8> {
        let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(level));
        encoder.write_all(payload).unwrap();
        encoder.finish().unwrap()
    }

    /// The decompressed KTLX 2013 trim: a real volume header, metadata
    /// record and radials.
    fn real_payload() -> Option<Vec<u8>> {
        let file = match recast_radar_testdata::bytes("l2-ktlx-20130520-201643-trim") {
            Ok(bytes) => bytes,
            Err(e) if e.is_offline() => {
                eprintln!("skipping: {e}");
                return None;
            }
            Err(e) => panic!("{e}"),
        };
        Some(crate::normalize_archive_bytes(&file).unwrap().0)
    }

    #[test]
    fn bit_writer_round_trips_bits_at() {
        let input: Vec<u8> = (0..64u8).map(|b| b.wrapping_mul(37)).collect();
        for start in [0usize, 1, 7, 8, 13, 100] {
            for end in [start + 1, start + 9, start + 57, 480] {
                if end > input.len() * 8 {
                    continue;
                }
                let stream = one_block_stream(&input, b'9', start, end, 0xdead_beef);
                let length = end - start;
                for bit in 0..length {
                    assert_eq!(
                        bits_at(&stream, 32 + bit, 1),
                        bits_at(&input, start + bit, 1),
                        "{start}..{end} bit {bit}"
                    );
                }
                assert_eq!(bits_at(&stream, 32 + length, 48), Some(END_MAGIC));
                assert_eq!(bits_at(&stream, 80 + length, 32), Some(0xdead_beef));
            }
        }
    }

    /// A real volume in a multi-block stream (100 kB blocks at level 1):
    /// every prefix length decodes to the payload's own prefix, only as far
    /// as needed, and a length past the end decodes it all.
    #[test]
    fn prefixes_of_a_multi_block_real_stream() {
        let Some(payload) = real_payload() else {
            return;
        };
        let stream = compress(&payload, 1);
        assert!(payload.len() > 1_000_000, "{} bytes", payload.len());
        for prefix_len in [1, 24, 99_999, 100_000, 100_001, 325_912, 777_777] {
            let (out, complete) = decode_prefix(&stream, prefix_len, "test").unwrap().unwrap();
            assert!(out.len() >= prefix_len, "{prefix_len}");
            assert!(
                out.len() < prefix_len + 1_000_000,
                "{prefix_len}: stops early"
            );
            assert!(!complete, "{prefix_len}");
            assert!(out == payload[..out.len()], "{prefix_len}: bytes differ");
        }
        let (out, complete) = decode_prefix(&stream, usize::MAX, "test").unwrap().unwrap();
        assert!(complete);
        assert!(out == payload, "whole stream");
    }

    /// Every magic at every bit offset, one bit at a time.
    fn magics_by_bit(input: &[u8]) -> Vec<(usize, u64)> {
        (0..(input.len() * 8).saturating_sub(MAGIC_BITS - 1))
            .filter_map(|bit| match bits_at(input, bit, MAGIC_BITS) {
                Some(magic @ (BLOCK_MAGIC | END_MAGIC)) => Some((bit, magic)),
                _ => None,
            })
            .collect()
    }

    /// The finder reports exactly the magics a bit-by-bit search finds, from
    /// every starting bit near each (real streams at levels 1 and 9, so the
    /// block boundaries fall at many bit offsets).
    #[test]
    fn magic_finder_matches_a_bit_by_bit_search() {
        let Some(payload) = real_payload() else {
            return;
        };
        let finder = MagicFinder::new();
        for level in [1, 9] {
            let stream = compress(&payload, level);
            let expected = magics_by_bit(&stream);
            assert!(expected.len() > 2, "level {level}: {expected:?}");
            let shifts: std::collections::BTreeSet<usize> =
                expected.iter().map(|(bit, _)| bit % 8).collect();
            if level == 1 {
                assert!(shifts.len() > 3, "block starts at bit shifts {shifts:?}");
            }
            let mut previous = 0;
            for &(bit, magic) in &expected {
                for from in [previous, bit.saturating_sub(40), bit.saturating_sub(1), bit] {
                    let from = from.max(previous);
                    assert_eq!(
                        finder.next_magic(&stream, from),
                        Some((bit, magic)),
                        "level {level} from {from}"
                    );
                }
                previous = bit + 1;
            }
            assert_eq!(finder.next_magic(&stream, previous), None);
        }
    }

    /// Every byte of `source` in order, the parser's way: extend, read,
    /// release. Returns the bytes and the largest window capacity seen.
    fn read_all(source: &mut Bzip2RecordBytes<'_>, step: usize) -> Result<(Vec<u8>, usize)> {
        let mut out = Vec::new();
        let mut widest = 0;
        loop {
            let start = out.len();
            let end = source.extend_to(start + step)?;
            out.extend_from_slice(source.get(start, end)?);
            source.release_before(end);
            widest = widest.max(source.window.capacity());
            if end < start + step {
                return Ok((out, widest));
            }
        }
    }

    /// The record source gives exactly the whole decode's bytes. With 100 kB
    /// blocks (level 1; a block's output is larger, because run-length
    /// coding comes before the block limit) it holds a few blocks' output
    /// rather than the 5.5 MB expanded stream; at level 9 this payload is
    /// only a few blocks long.
    #[test]
    fn record_bytes_decode_block_by_block() {
        let Some(payload) = real_payload() else {
            return;
        };
        for level in [1, 9] {
            let stream = compress(&payload, level);
            let mut source = Bzip2RecordBytes::new(&stream, usize::MAX, "test");
            let (out, widest) = read_all(&mut source, 2432).unwrap();
            assert!(out == payload, "level {level}: bytes differ");
            assert!(source.done && source.error.is_none());
            if level == 1 {
                assert!(
                    widest < payload.len() / 4,
                    "window grew to {widest} for {} bytes",
                    payload.len()
                );
                // Released bytes were dropped: asking for them again is an
                // error, not a panic.
                assert!(source.base > 0);
                assert!(matches!(
                    source.get(0, 1),
                    Err(NexradError::InvalidMessage { offset: 0, .. })
                ));
            }
            assert!(source.get(0, source.produced() + 1).is_err());
            assert!(source.finish_after(Ok(())).is_ok());
        }
    }

    /// When the block-wise decode gives up part-way (as after a boundary
    /// found by mistake), the rest comes from the whole decode and the bytes
    /// are the same.
    #[test]
    fn record_bytes_fall_back_to_the_whole_decode() {
        let Some(payload) = real_payload() else {
            return;
        };
        let stream = compress(&payload, 1);
        let mut source = Bzip2RecordBytes::new(&stream, usize::MAX, "test");
        let first = source.extend_to(250_000).unwrap();
        assert_eq!(first, 250_000);
        let head = source.get(0, first).unwrap().to_vec();
        source.release_before(120_000);
        source.decode_whole();
        let rest_end = source.extend_to(usize::MAX).unwrap();
        assert_eq!(rest_end, payload.len());
        assert!(head == payload[..first]);
        assert!(*source.get(120_000, rest_end).unwrap() == payload[120_000..]);
        assert!(source.finish_after(Ok(())).is_ok());
    }

    /// Damage and the output limit fail with the whole decode's error, even
    /// when the parser stopped reading before the damaged block.
    #[test]
    fn record_bytes_fail_like_the_whole_decode() {
        let Some(payload) = real_payload() else {
            return;
        };
        let stream = compress(&payload, 1);
        let one_shot = |input: &[u8], limit: usize| {
            let mut out = Vec::new();
            match decompress_bzip2_stream_into(input, &mut out, limit, "test") {
                Err(NexradError::Compression(message)) => message,
                other => panic!("{other:?}"),
            }
        };
        let mut damaged = stream.clone();
        let len = damaged.len();
        damaged[len * 7 / 8] ^= 0x55;
        let expected = one_shot(&damaged, usize::MAX);

        let mut source = Bzip2RecordBytes::new(&damaged, usize::MAX, "test");
        let error = read_all(&mut source, 2432).unwrap_err();
        assert!(
            matches!(&error, NexradError::Compression(m) if *m == expected),
            "{error}"
        );

        let mut early_stop = Bzip2RecordBytes::new(&damaged, usize::MAX, "test");
        assert_eq!(early_stop.extend_to(100).unwrap(), 100);
        let error = early_stop.finish_after(Ok(())).unwrap_err();
        assert!(
            matches!(&error, NexradError::Compression(m) if *m == expected),
            "{error}"
        );

        let limit = payload.len() / 2;
        let expected = one_shot(&stream, limit);
        let mut limited = Bzip2RecordBytes::new(&stream, limit, "test");
        let error = read_all(&mut limited, 2432).unwrap_err();
        assert!(
            matches!(&error, NexradError::Compression(m) if *m == expected),
            "{error}"
        );
    }

    /// Damage in a later block does not affect a prefix that ends before it;
    /// damage in the first block makes the block-wise decode give up (the
    /// caller then decodes the stream whole and reports the error).
    #[test]
    fn damage_after_the_prefix_is_not_decoded() {
        let Some(payload) = real_payload() else {
            return;
        };
        let stream = compress(&payload, 1);
        let mut late = stream.clone();
        let len = late.len();
        late[len * 3 / 4] ^= 0x55;
        let (out, _) = decode_prefix(&late, 100_000, "test").unwrap().unwrap();
        assert!(out == payload[..out.len()]);

        let mut early = stream.clone();
        early[40] ^= 0x55;
        assert!(decode_prefix(&early, 100_000, "test").unwrap().is_none());
    }
}
