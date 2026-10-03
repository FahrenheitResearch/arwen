//! gzip inflate for `.gz` Archive II volumes: a sliding window the record
//! parser reads while it inflates (`GzipRecordBytes`, the volume decode),
//! a whole-buffer one-shot for callers that want the expanded bytes, and a
//! streaming reader for the preview paths.
//!
//! Both decode every member of a multi-member gzip file and concatenate the
//! outputs, as `gzip -d` does, and both ignore bytes after the last member
//! that do not start another one (zero padding, for example). A
//! `flate2::read::GzDecoder` stops after the first member, which turned a
//! re-gzipped volume with several members into a silently partial volume;
//! `flate2::read::MultiGzDecoder` decodes every member but errors on
//! trailing bytes.
//!
//! The one-shot exists for speed. Reading a `GzDecoder` through a 64 KiB
//! chunk loop inflates in small output windows: every call re-enters
//! zlib-rs, which then copies the newest 32 KiB into its sliding window, and
//! every chunk is copied a second time into a growing `Vec`. When the whole
//! compressed file is in memory, zlib-rs can write straight into one output
//! buffer sized from the gzip ISIZE trailer, as zlib's `uncompress` does. On
//! the 9.5 MB KTLX20130520 volume that removes about a sixth of the decode's
//! instructions and about 5% of its wall time. The volume decode goes one
//! step further: `GzipRecordBytes` inflates the same way into a reused
//! window of about `WINDOW_CHUNK` (1 MiB) bytes that the parser consumes, so the
//! expanded volume (45 MB for KTLX20130520) is neither held beside the
//! decoded one nor zero-filled before inflating.

use std::io::{self, BufRead, Read};

use flate2::bufread::GzDecoder;
use flate2::{Decompress, FlushDecompress, Status};

use crate::{NexradError, RecordBytes};

/// gzip member magic bytes (RFC 1952 ID1, ID2).
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Smallest possible gzip member: 10-byte header, a 2-byte empty final
/// fixed-Huffman block, and the 8-byte CRC-32 + ISIZE trailer.
const GZIP_MIN_MEMBER_LEN: usize = 20;

/// Deflate cannot expand input by more than about 1032:1 (a 258-byte match
/// costs at least 2 bits), so a trailer that claims more than this is not
/// trusted as an allocation size.
const MAX_DEFLATE_EXPANSION: usize = 1032;

/// Minimum growth step when the ISIZE hint turns out to be too small
/// (multi-member files, or a wrong or wrapped trailer).
const MIN_GROWTH: usize = 64 * 1024;

/// Output room given to each inflate call of [`GzipRecordBytes`]. zlib-rs
/// keeps its own copy of the last 32 KiB when the output moves between
/// calls, so a call costs a 32 KiB copy; 1 MiB windows keep that near 3%.
pub(crate) const WINDOW_CHUNK: usize = 1 << 20;

/// Inflate every gzip member in `raw` into one buffer of at most `limit`
/// bytes.
///
/// The output buffer is presized from the ISIZE field of the final trailer.
/// For a single-member file (every gzip-compressed Archive II volume seen so
/// far) that is the exact length, so the whole stream inflates in one
/// zlib-rs call with no chunk copies and no reallocation. ISIZE only
/// describes the *last* member, modulo 2^32. For a multi-member file the
/// hint is only a starting size, and the buffer grows when the output fills
/// it.
///
/// Members are decoded back to back and their outputs are concatenated, as
/// `gzip -d` and `flate2::read::MultiGzDecoder` do. zlib-rs checks each
/// member's header, CRC-32 and ISIZE. Once a member ends, any remaining bytes
/// that do not begin with the gzip magic are ignored, as
/// `flate2::read::GzDecoder` ignores data after its member (for example
/// zero padding).
///
/// Errors match `bounded_read::read_to_end_limited`: exactly `limit` bytes
/// is accepted, one more is "expands beyond the limit", and an allocation
/// failure is reported instead of aborting. The message is the complete
/// diagnostic, for the caller's compression-error variant.
pub fn inflate_gzip_members_limited(
    raw: &[u8],
    limit: usize,
    context: &'static str,
) -> Result<Vec<u8>, String> {
    // One byte of room past `limit` tells "ends exactly at the limit" apart
    // from "expands beyond it".
    let max_len = limit.saturating_add(1);
    let mut output = Vec::new();
    resize_zeroed(&mut output, gzip_size_hint(raw, limit), context)?;
    let mut filled = 0usize;
    let mut input = raw;
    loop {
        // One member: header, deflate body, CRC-32 + ISIZE trailer, all
        // parsed and verified by zlib-rs in gzip mode.
        let mut inflater = Decompress::new_gzip(15);
        loop {
            if filled == output.len() {
                let grown = output
                    .len()
                    .saturating_add(output.len().max(MIN_GROWTH))
                    .min(max_len);
                resize_zeroed(&mut output, grown, context)?;
            }
            let in_before = inflater.total_in();
            let out_before = inflater.total_out();
            let status = inflater
                .decompress(input, &mut output[filled..], FlushDecompress::Finish)
                .map_err(|err| format!("{context}: {err}"))?;
            // Both deltas are bounded by the lengths of the slices just
            // passed in, so they fit in usize.
            let consumed = (inflater.total_in() - in_before) as usize;
            input = &input[consumed..];
            filled += (inflater.total_out() - out_before) as usize;
            if filled > limit {
                return Err(format!("{context} expands beyond the {limit}-byte limit"));
            }
            match status {
                Status::StreamEnd => break,
                // With all remaining input supplied and `Finish`, inflate
                // stops before the end of the member only when it runs out
                // of output space or input. If the output is full, grow it
                // and continue. zlib-rs has saved its window, so the next
                // slice does not need to be contiguous with this one.
                _ if filled == output.len() => {}
                // zlib-rs takes at most u32::MAX input bytes per call, so
                // keep going while input remains and the call made progress.
                _ if consumed != 0 && !input.is_empty() => {}
                _ => return Err(format!("{context}: truncated gzip stream")),
            }
        }
        if !input.starts_with(&GZIP_MAGIC) {
            break;
        }
    }
    output.truncate(filled);
    Ok(output)
}

/// Starting output size: the final ISIZE field when it is plausible for
/// this input and within `limit`, otherwise 0 (grow on demand).
fn gzip_size_hint(raw: &[u8], limit: usize) -> usize {
    if raw.len() < GZIP_MIN_MEMBER_LEN {
        return 0;
    }
    let [.., b0, b1, b2, b3] = *raw else {
        return 0;
    };
    let trailer_len = usize::try_from(u32::from_le_bytes([b0, b1, b2, b3])).unwrap_or(usize::MAX);
    let plausible = raw.len().saturating_mul(MAX_DEFLATE_EXPANSION).min(limit);
    if trailer_len <= plausible {
        trailer_len
    } else {
        0
    }
}

/// Grow `output` to `len` zeroed bytes, reporting allocation failure as an
/// error instead of aborting.
fn resize_zeroed(output: &mut Vec<u8>, len: usize, context: &'static str) -> Result<(), String> {
    output
        .try_reserve_exact(len.saturating_sub(output.len()))
        .map_err(|err| format!("{context}: cannot reserve decoded buffer: {err}"))?;
    output.resize(len, 0);
    Ok(())
}

/// A gzip file inflated on demand for the record parser: the bytes the
/// parser has not released, in one window that inflate calls write into.
///
/// It inflates every member as [`inflate_gzip_members_limited`] does, with
/// the same checks (header, CRC-32 and ISIZE of each member by zlib-rs, the
/// output limit, truncation, trailing bytes that are not another member are
/// ignored) and the same error messages, but keeps only about
/// [`WINDOW_CHUNK`] bytes plus the message being parsed. Before each inflate
/// call the released prefix is dropped, which moves only the partly read
/// message, and the window's memory is reused: it is zero-filled once when
/// it grows, not once per inflated byte. [`Self::finish_after`] inflates
/// whatever the parser did not read, so a volume decodes exactly when its
/// one-shot inflate would have succeeded.
pub(crate) struct GzipRecordBytes<'a> {
    /// Compressed input not yet consumed.
    input: &'a [u8],
    inflater: Decompress,
    /// `window[..filled]` holds decoded bytes `base..base + filled`; the
    /// rest is room for the next inflate call.
    window: Vec<u8>,
    filled: usize,
    base: usize,
    /// Offset before which the parser reads nothing again.
    released: usize,
    /// No more output: the last member ended, or `error` is set.
    done: bool,
    /// The first inflate error, returned again by later calls.
    error: Option<String>,
    limit: usize,
    context: &'static str,
}

impl<'a> GzipRecordBytes<'a> {
    pub(crate) fn new(raw: &'a [u8], limit: usize, context: &'static str) -> Self {
        Self {
            input: raw,
            inflater: Decompress::new_gzip(15),
            window: Vec::new(),
            filled: 0,
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
        self.base + self.filled
    }

    /// One inflate call into the window's room.
    fn inflate_more(&mut self) -> Result<(), String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        let result = self.inflate_call();
        if let Err(error) = &result {
            self.done = true;
            self.error = Some(error.clone());
        }
        result
    }

    fn inflate_call(&mut self) -> Result<(), String> {
        let context = self.context;
        // Drop the released prefix first. Inflating is only needed when the
        // parser wants bytes past the window, so what remains is the part of
        // one message already inflated: moving it is cheap.
        let release = self.released.saturating_sub(self.base).min(self.filled);
        if release > 0 && release >= self.filled / 2 {
            self.window.copy_within(release..self.filled, 0);
            self.filled -= release;
            self.base += release;
        }
        if self.window.len() - self.filled < WINDOW_CHUNK / 2 {
            let grown = self.filled.saturating_add(WINDOW_CHUNK);
            resize_zeroed(&mut self.window, grown, context)?;
        }
        let in_before = self.inflater.total_in();
        let out_before = self.inflater.total_out();
        let status = self
            .inflater
            .decompress(
                self.input,
                &mut self.window[self.filled..],
                FlushDecompress::Finish,
            )
            .map_err(|err| format!("{context}: {err}"))?;
        // Both deltas are bounded by the lengths of the slices just passed
        // in, so they fit in usize.
        let consumed = (self.inflater.total_in() - in_before) as usize;
        self.input = &self.input[consumed..];
        self.filled += (self.inflater.total_out() - out_before) as usize;
        if self.produced() > self.limit {
            let limit = self.limit;
            return Err(format!("{context} expands beyond the {limit}-byte limit"));
        }
        match status {
            Status::StreamEnd => {
                if self.input.starts_with(&GZIP_MAGIC) {
                    self.inflater = Decompress::new_gzip(15);
                } else {
                    self.done = true;
                }
            }
            // Out of output room (the next call makes more), or input left
            // and progress made: keep going, as the one-shot does.
            _ if self.filled == self.window.len() => {}
            _ if consumed != 0 && !self.input.is_empty() => {}
            _ => return Err(format!("{context}: truncated gzip stream")),
        }
        Ok(())
    }

    /// Inflate the rest of the file, discarding it, and return `parsed`
    /// unless the file fails a gzip check, whose error then wins (the
    /// one-shot inflate reported it before any parsing).
    pub(crate) fn finish_after<T>(mut self, parsed: crate::Result<T>) -> crate::Result<T> {
        while !self.done {
            self.base += self.filled;
            self.released = self.base;
            self.filled = 0;
            if let Err(error) = self.inflate_more() {
                return Err(NexradError::Compression(error));
            }
        }
        match self.error {
            Some(error) => Err(NexradError::Compression(error)),
            None => parsed,
        }
    }
}

impl RecordBytes for GzipRecordBytes<'_> {
    fn extend_to(&mut self, end: usize) -> crate::Result<usize> {
        while self.produced() < end && !self.done {
            self.inflate_more().map_err(NexradError::Compression)?;
        }
        Ok(end.min(self.produced()))
    }

    fn get(&self, start: usize, end: usize) -> crate::Result<&[u8]> {
        crate::window_range(&self.window[..self.filled], self.base, start, end)
    }

    fn release_before(&mut self, offset: usize) {
        self.released = self.released.max(offset);
    }
}

/// Streaming reader over every member of a gzip file, for the preview
/// decoders that stop before the end of the input.
///
/// Each member is read through a `flate2::bufread::GzDecoder` (header,
/// CRC-32 and ISIZE checked). When a member ends, the bytes that follow are
/// inspected: another gzip magic starts the next member, anything else ends
/// the stream and is ignored, so the reader agrees with
/// [`inflate_gzip_members_limited`] on the same input. Inflating stops as
/// soon as the caller stops reading.
pub(crate) struct MultiGzReader<R: BufRead> {
    /// The member being read; `None` once the input is exhausted or a
    /// non-member tail was found.
    member: Option<GzDecoder<R>>,
}

impl<R: BufRead> MultiGzReader<R> {
    pub(crate) fn new(reader: R) -> Self {
        Self {
            member: Some(GzDecoder::new(reader)),
        }
    }
}

impl<R: BufRead> Read for MultiGzReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while let Some(member) = self.member.as_mut() {
            let count = member.read(buf)?;
            if count > 0 || buf.is_empty() {
                return Ok(count);
            }
            // The member is finished and exactly its bytes were consumed
            // (`bufread::GzDecoder` reads the deflate body and the 8-byte
            // trailer through the inner `BufRead` without reading ahead).
            let Some(member) = self.member.take() else {
                break;
            };
            let mut inner = member.into_inner();
            // A one-byte fill cannot show the second magic byte; the header
            // parser of the next decoder decides that case.
            let next_member = matches!(inner.fill_buf()?, [0x1f, 0x8b, ..] | [0x1f]);
            if next_member {
                self.member = Some(GzDecoder::new(inner));
            }
        }
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::read::{GzDecoder, MultiGzDecoder};
    use flate2::write::GzEncoder;
    use std::io::Write;

    const LIMIT: usize = 512 * 1024 * 1024;
    const CTX: &str = "gzip test payload";

    fn gzip(payload: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(payload).unwrap();
        encoder.finish().unwrap()
    }

    /// The KTLX 1999-05-04 volume as archived: a whole-file gzip object
    /// (`ARCHIVE2` header, Message 1 radials in raw 2432-byte records),
    /// downloaded on first use.
    const KTLX_1999_GZIP: &str = "l2-ktlx-19990504-002218";
    /// The committed trim of the same volume (LDM bzip2 records): its
    /// decompressed bytes are the real payload the gzip cases wrap.
    const KTLX_1999_TRIM: &str = "l2-ktlx-19990504-002218-trim";

    /// The real gzip archive, or `None` when it is not cached and cannot be
    /// downloaded now.
    fn real_gzip_archive() -> Option<Vec<u8>> {
        match recast_radar_testdata::bytes(KTLX_1999_GZIP) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.is_offline() => {
                eprintln!("skipping {KTLX_1999_GZIP}: {e}");
                None
            }
            Err(e) => panic!("{e}"),
        }
    }

    /// `len` bytes of the decompressed real trim (volume header, metadata
    /// messages, Message 1 radials), starting where `seed` points, so
    /// different seeds give different stretches of radial data.
    fn payload(len: usize, seed: u32) -> Vec<u8> {
        use std::sync::OnceLock;
        static DECODED: OnceLock<Vec<u8>> = OnceLock::new();
        let decoded = DECODED.get_or_init(|| {
            let trim =
                recast_radar_testdata::bytes(KTLX_1999_TRIM).unwrap_or_else(|e| panic!("{e}"));
            let (decoded, _) = crate::normalize_archive_bytes(&trim).unwrap();
            assert!(decoded.starts_with(b"ARCHIVE2"));
            decoded
        });
        assert!(
            len <= decoded.len(),
            "{len} bytes asked of a {}-byte archive",
            decoded.len()
        );
        let offset = (seed as usize).wrapping_mul(61_803) % (decoded.len() - len + 1);
        decoded[offset..offset + len].to_vec()
    }

    fn multi_gz_reference(raw: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        MultiGzDecoder::new(raw).read_to_end(&mut out).unwrap();
        out
    }

    /// The streaming reader's output, read through small buffers so member
    /// boundaries fall inside reads.
    fn streamed(raw: &[u8]) -> io::Result<Vec<u8>> {
        let mut reader = MultiGzReader::new(raw);
        let mut out = Vec::new();
        let mut chunk = [0u8; 4093];
        loop {
            let count = reader.read(&mut chunk)?;
            if count == 0 {
                return Ok(out);
            }
            out.extend_from_slice(&chunk[..count]);
        }
    }

    #[test]
    fn real_gzip_archive_matches_gz_decoder_and_presizes_exactly() {
        let Some(raw) = real_gzip_archive() else {
            return;
        };
        assert!(raw.starts_with(&[0x1f, 0x8b]));
        let out = inflate_gzip_members_limited(&raw, LIMIT, CTX).unwrap();
        let mut reference = Vec::new();
        GzDecoder::new(raw.as_slice())
            .read_to_end(&mut reference)
            .unwrap();
        assert_eq!(out, reference);
        assert!(out.starts_with(b"ARCHIVE2"));
        assert_eq!(gzip_size_hint(&raw, LIMIT), out.len());
        assert_eq!(out.capacity(), out.len());
        assert_eq!(streamed(&raw).unwrap(), out);
    }

    /// The record source gives exactly the one-shot inflate's bytes, read
    /// the parser's way (extend, read, release) through a window much
    /// smaller than the archive; released bytes are an error to ask for
    /// again, not a panic.
    #[test]
    fn record_bytes_match_the_one_shot_inflate() {
        let Some(raw) = real_gzip_archive() else {
            return;
        };
        let data = inflate_gzip_members_limited(&raw, LIMIT, CTX).unwrap();
        assert!(data.len() > 4 * WINDOW_CHUNK, "{} bytes", data.len());
        let mut source = GzipRecordBytes::new(&raw, LIMIT, CTX);
        let mut out = Vec::new();
        let step = 2432;
        loop {
            let start = out.len();
            let end = source.extend_to(start + step).unwrap();
            out.extend_from_slice(source.get(start, end).unwrap());
            source.release_before(end);
            assert!(source.window.len() <= 2 * WINDOW_CHUNK);
            if end < start + step {
                break;
            }
        }
        assert!(out == data);
        assert!(source.base > 0);
        assert!(matches!(
            source.get(0, 1),
            Err(NexradError::InvalidMessage { offset: 0, .. })
        ));
        assert!(source.get(out.len(), out.len() + 1).is_err());
        assert!(source.finish_after(Ok(())).is_ok());
    }

    #[test]
    fn single_member_matches_gz_decoder_and_presizes_exactly() {
        let data = payload(300_000, 1);
        let raw = gzip(&data);
        let out = inflate_gzip_members_limited(&raw, LIMIT, CTX).unwrap();
        let mut reference = Vec::new();
        GzDecoder::new(raw.as_slice())
            .read_to_end(&mut reference)
            .unwrap();
        assert_eq!(out, reference);
        assert_eq!(out, data);
        assert_eq!(gzip_size_hint(&raw, LIMIT), data.len());
        // The ISIZE hint was exact: no growth step happened.
        assert_eq!(out.capacity(), data.len());
        assert_eq!(streamed(&raw).unwrap(), data);
    }

    #[test]
    fn empty_payload_decodes_to_empty() {
        let raw = gzip(&[]);
        assert!(
            inflate_gzip_members_limited(&raw, LIMIT, CTX)
                .unwrap()
                .is_empty()
        );
        assert!(streamed(&raw).unwrap().is_empty());
    }

    #[test]
    fn multi_member_decodes_every_member() {
        let parts = [payload(200_000, 2), payload(150_000, 3), payload(10, 4)];
        let mut raw = Vec::new();
        for part in &parts {
            raw.extend(gzip(part));
        }
        let expected = parts.concat();
        // The last member's ISIZE (10) is far below the total, so this
        // exercises the growth path.
        assert_eq!(gzip_size_hint(&raw, LIMIT), 10);
        let out = inflate_gzip_members_limited(&raw, LIMIT, CTX).unwrap();
        assert_eq!(out, expected);
        assert_eq!(out, multi_gz_reference(&raw));
        assert_eq!(streamed(&raw).unwrap(), expected);
    }

    #[test]
    fn multi_member_with_empty_members() {
        let first = payload(90_000, 5);
        let mut raw = gzip(&[]);
        raw.extend(gzip(&first));
        raw.extend(gzip(&[]));
        assert_eq!(gzip_size_hint(&raw, LIMIT), 0);
        let out = inflate_gzip_members_limited(&raw, LIMIT, CTX).unwrap();
        assert_eq!(out, first);
        assert_eq!(out, multi_gz_reference(&raw));
        assert_eq!(streamed(&raw).unwrap(), first);
    }

    #[test]
    fn hint_from_last_member_grows_once_for_earlier_members() {
        // ISIZE describes only the large last member, so the buffer fills
        // while that member is still inflating and grows once.
        let parts = [payload(5_000, 6), payload(400_000, 7)];
        let mut raw = gzip(&parts[0]);
        raw.extend(gzip(&parts[1]));
        assert_eq!(gzip_size_hint(&raw, LIMIT), 400_000);
        let out = inflate_gzip_members_limited(&raw, LIMIT, CTX).unwrap();
        assert_eq!(out, parts.concat());
    }

    #[test]
    fn trailing_non_gzip_bytes_are_ignored_like_gz_decoder() {
        let data = payload(70_000, 8);
        let mut raw = gzip(&data);
        raw.extend_from_slice(&[0u8; 512]);
        let out = inflate_gzip_members_limited(&raw, LIMIT, CTX).unwrap();
        assert_eq!(out, data);
        assert_eq!(streamed(&raw).unwrap(), data);

        let mut raw = gzip(&data);
        raw.extend_from_slice(b"not gzip");
        let out = inflate_gzip_members_limited(&raw, LIMIT, CTX).unwrap();
        assert_eq!(out, data);
        assert_eq!(streamed(&raw).unwrap(), data);
    }

    #[test]
    fn truncated_member_is_an_error() {
        let data = payload(120_000, 9);
        let raw = gzip(&data);
        for cut in [raw.len() - 1, raw.len() - 8, raw.len() / 2, 12] {
            let err = inflate_gzip_members_limited(&raw[..cut], LIMIT, CTX).unwrap_err();
            assert!(err.starts_with(CTX), "{cut}: {err}");
            assert!(streamed(&raw[..cut]).is_err(), "streamed {cut}");
        }
        // A truncated second member is an error too.
        let mut two = raw.clone();
        two.extend_from_slice(&raw[..raw.len() / 3]);
        assert!(inflate_gzip_members_limited(&two, LIMIT, CTX).is_err());
        assert!(streamed(&two).is_err());
    }

    #[test]
    fn corrupt_crc_or_isize_is_an_error() {
        let data = payload(50_000, 10);
        let raw = gzip(&data);
        let n = raw.len();
        let mut bad_crc = raw.clone();
        bad_crc[n - 8] ^= 1;
        assert!(inflate_gzip_members_limited(&bad_crc, LIMIT, CTX).is_err());
        assert!(streamed(&bad_crc).is_err());
        for bad_len in [data.len() as u32 - 1, data.len() as u32 + 1, 0, u32::MAX] {
            let mut bad_isize = raw.clone();
            bad_isize[n - 4..].copy_from_slice(&bad_len.to_le_bytes());
            assert!(
                inflate_gzip_members_limited(&bad_isize, LIMIT, CTX).is_err(),
                "ISIZE {bad_len}"
            );
            assert!(streamed(&bad_isize).is_err(), "streamed ISIZE {bad_len}");
        }
        let mut bad_body = raw.clone();
        bad_body[n / 2] ^= 0x55;
        assert!(inflate_gzip_members_limited(&bad_body, LIMIT, CTX).is_err());
        assert!(streamed(&bad_body).is_err());
    }

    #[test]
    fn limit_accepts_exact_size_and_rejects_one_more_byte() {
        let data = payload(100_000, 11);
        let raw = gzip(&data);
        assert_eq!(
            inflate_gzip_members_limited(&raw, data.len(), CTX).unwrap(),
            data
        );
        let err = inflate_gzip_members_limited(&raw, data.len() - 1, CTX).unwrap_err();
        assert!(err.contains("expands beyond"), "{err}");
        // Multi-member output is limited in aggregate.
        let mut two = raw.clone();
        two.extend(gzip(&data));
        let err = inflate_gzip_members_limited(&two, data.len() * 2 - 1, CTX).unwrap_err();
        assert!(err.contains("expands beyond"), "{err}");
        assert_eq!(
            inflate_gzip_members_limited(&two, data.len() * 2, CTX)
                .unwrap()
                .len(),
            data.len() * 2
        );
    }

    #[test]
    fn size_hint_ignores_implausible_or_over_limit_trailers() {
        let raw = gzip(&payload(1_000, 12));
        let n = raw.len();
        let mut huge = raw.clone();
        huge[n - 4..].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(gzip_size_hint(&huge, LIMIT), 0);
        assert_eq!(gzip_size_hint(&raw, 999), 0);
        assert_eq!(gzip_size_hint(&raw, 1_000), 1_000);
        assert_eq!(gzip_size_hint(&raw[..GZIP_MIN_MEMBER_LEN - 1], LIMIT), 0);
    }

    #[test]
    fn streaming_reader_handles_one_byte_fills() {
        // A reader that hands out one byte per fill: the magic of every
        // member after the first is seen one byte at a time.
        struct OneByte<'a>(&'a [u8]);
        impl Read for OneByte<'_> {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                let count = buf.len().min(self.0.len()).min(1);
                buf[..count].copy_from_slice(&self.0[..count]);
                self.0 = &self.0[count..];
                Ok(count)
            }
        }
        impl BufRead for OneByte<'_> {
            fn fill_buf(&mut self) -> io::Result<&[u8]> {
                Ok(&self.0[..self.0.len().min(1)])
            }
            fn consume(&mut self, amt: usize) {
                self.0 = &self.0[amt..];
            }
        }
        let parts = [payload(30_000, 13), payload(20_000, 14)];
        let mut raw = gzip(&parts[0]);
        raw.extend(gzip(&parts[1]));
        let mut out = Vec::new();
        MultiGzReader::new(OneByte(&raw))
            .read_to_end(&mut out)
            .unwrap();
        assert_eq!(out, parts.concat());
        // Trailing zero padding after the last member is ignored.
        raw.extend_from_slice(&[0u8; 100]);
        let mut out = Vec::new();
        MultiGzReader::new(OneByte(&raw))
            .read_to_end(&mut out)
            .unwrap();
        assert_eq!(out, parts.concat());
    }
}
