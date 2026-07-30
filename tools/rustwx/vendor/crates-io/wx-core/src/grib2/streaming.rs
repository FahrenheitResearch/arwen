//! Incremental GRIB2 parser that processes messages as bytes arrive.
//!
//! The key insight: each GRIB2 message starts with the 4-byte magic `GRIB`,
//! followed by 2 reserved bytes, 1 byte discipline, 1 byte edition, then an
//! 8-byte big-endian total message length (bytes 8..16 of the indicator section).
//!
//! This means we can parse messages incrementally:
//! 1. Scan for `GRIB` magic in the buffer
//! 2. Read the 8-byte total length from bytes 8..16
//! 3. Wait until we have that many bytes from the start of the magic
//! 4. Parse the complete message using the existing `Grib2File::from_bytes`
//! 5. Drain the consumed bytes and continue scanning

use super::parser::{Grib2File, Grib2Message};

/// Minimum size of a GRIB2 indicator section (Section 0).
/// 4 (magic) + 2 (reserved) + 1 (discipline) + 1 (edition) + 8 (total length) = 16 bytes.
const INDICATOR_SIZE: usize = 16;

/// Incremental GRIB2 parser that processes messages as they arrive.
///
/// Feed bytes from a network stream (or any source) via [`Self::feed`], and
/// extract fully-parsed messages via [`Self::take_messages`].
///
/// # Example
///
/// ```
/// use wx_core::grib2::streaming::StreamingParser;
///
/// let mut parser = StreamingParser::new();
///
/// // In a real scenario, `chunk` comes from a network read
/// // parser.feed(&chunk);
///
/// // Check if any complete messages are ready
/// if parser.has_messages() {
///     let messages = parser.take_messages();
///     for msg in &messages {
///         println!("Got message: discipline={}", msg.discipline);
///     }
/// }
/// ```
pub struct StreamingParser {
    /// Accumulation buffer for incoming bytes.
    buffer: Vec<u8>,
    /// Fully parsed messages waiting to be consumed.
    messages: Vec<Grib2Message>,
    /// Number of leading bytes in `buffer` that have been scanned and found
    /// to contain no GRIB magic. We skip re-scanning these on subsequent
    /// `feed` calls for efficiency.
    scan_start: usize,
    /// Total number of messages parsed over the lifetime of this parser.
    total_parsed: usize,
}

impl StreamingParser {
    /// Create a new empty streaming parser.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            messages: Vec::new(),
            scan_start: 0,
            total_parsed: 0,
        }
    }

    /// Create a new streaming parser with a pre-allocated buffer capacity.
    ///
    /// Use this when you know the approximate total download size to avoid
    /// repeated allocations.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            messages: Vec::new(),
            scan_start: 0,
            total_parsed: 0,
        }
    }

    /// Feed more bytes into the parser. Returns the number of complete
    /// messages parsed so far (cumulative across all `feed` calls).
    ///
    /// This method eagerly parses as many complete messages as possible
    /// from the buffer after appending the new data.
    pub fn feed(&mut self, data: &[u8]) -> usize {
        if data.is_empty() {
            return self.total_parsed;
        }

        self.buffer.extend_from_slice(data);
        self.try_parse();
        self.total_parsed
    }

    /// Take all fully parsed messages out of the parser, leaving any
    /// partial/incomplete data in the internal buffer for future `feed` calls.
    ///
    /// After calling this, `has_messages()` will return `false` until more
    /// data is fed and new messages are parsed.
    pub fn take_messages(&mut self) -> Vec<Grib2Message> {
        std::mem::take(&mut self.messages)
    }

    /// Check if any complete messages are available for consumption.
    pub fn has_messages(&self) -> bool {
        !self.messages.is_empty()
    }

    /// Number of complete messages waiting to be taken.
    pub fn pending_count(&self) -> usize {
        self.messages.len()
    }

    /// Total number of messages parsed over the lifetime of this parser.
    pub fn total_parsed(&self) -> usize {
        self.total_parsed
    }

    /// Number of bytes currently buffered (not yet consumed by a complete message).
    pub fn buffered_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Finish parsing. Returns all remaining parsed messages.
    ///
    /// Returns an error if there are leftover bytes that look like a truncated
    /// GRIB2 message (i.e., a `GRIB` magic was found but not enough bytes
    /// followed to complete the message).
    pub fn finish(mut self) -> Result<Vec<Grib2Message>, String> {
        // One final parse attempt
        self.try_parse();

        // Check for leftover data that contains a partial message
        if !self.buffer.is_empty() {
            if let Some(magic_pos) = find_magic_in(&self.buffer, 0) {
                let remaining = self.buffer.len() - magic_pos;
                if remaining >= INDICATOR_SIZE {
                    // We can read the expected length
                    let expected = read_u64_be(&self.buffer, magic_pos + 8) as usize;
                    return Err(format!(
                        "Incomplete GRIB2 message: have {} bytes but message declares {} bytes",
                        remaining, expected
                    ));
                } else {
                    return Err(format!(
                        "Incomplete GRIB2 indicator: have {} bytes, need at least {}",
                        remaining, INDICATOR_SIZE
                    ));
                }
            }
            // Remaining bytes don't contain a GRIB magic — could be trailing
            // junk or padding. This is not an error for GRIB2 files.
        }

        Ok(self.messages)
    }

    /// Internal: try to parse as many complete messages as possible from the buffer.
    fn try_parse(&mut self) {
        loop {
            // Find the next GRIB magic starting from scan_start
            let magic_pos = match find_magic_in(&self.buffer, self.scan_start) {
                Some(pos) => pos,
                None => {
                    // No magic found. We can discard everything except the last
                    // 3 bytes (which could be a partial "GRIB" straddling a chunk
                    // boundary).
                    let keep = self.buffer.len().min(3);
                    let discard = self.buffer.len() - keep;
                    if discard > 0 {
                        self.buffer.drain(..discard);
                        self.scan_start = 0;
                    }
                    return;
                }
            };

            // Do we have enough bytes for the indicator section?
            if magic_pos + INDICATOR_SIZE > self.buffer.len() {
                // Not enough data yet — remember where we found magic so we
                // don't re-scan prefix bytes on next feed.
                self.scan_start = magic_pos;
                return;
            }

            // Read the edition byte to confirm this is GRIB2
            let edition = self.buffer[magic_pos + 7];
            if edition != 2 {
                // Not a GRIB2 message — skip past this false "GRIB" magic
                self.scan_start = magic_pos + 4;
                continue;
            }

            // Read total message length (bytes 8..16 of the message)
            let total_len = read_u64_be(&self.buffer, magic_pos + 8) as usize;

            // Sanity check: minimum GRIB2 message is 16 (indicator) + 4 (end "7777") = 20 bytes
            if total_len < 20 {
                // Bogus length — skip this magic
                self.scan_start = magic_pos + 4;
                continue;
            }

            // Do we have the full message?
            if magic_pos + total_len > self.buffer.len() {
                // Not yet — wait for more data
                self.scan_start = magic_pos;
                return;
            }

            // Extract the message bytes and parse
            let msg_bytes = &self.buffer[magic_pos..magic_pos + total_len];
            match Grib2File::from_bytes(msg_bytes) {
                Ok(grib) => {
                    self.messages.extend(grib.messages);
                    self.total_parsed += 1;
                }
                Err(e) => {
                    // Log parse error but continue — skip this message
                    eprintln!(
                        "StreamingParser: failed to parse message at offset {}: {}",
                        magic_pos, e
                    );
                }
            }

            // Drain consumed bytes (everything up to and including this message)
            let consumed = magic_pos + total_len;
            self.buffer.drain(..consumed);
            self.scan_start = 0;
        }
    }
}

impl Default for StreamingParser {
    fn default() -> Self {
        Self::new()
    }
}

// ─── helper functions ───────────────────────────────────────────────

/// Scan for the 4-byte "GRIB" magic starting at `start`.
fn find_magic_in(data: &[u8], start: usize) -> Option<usize> {
    if data.len() < 4 {
        return None;
    }
    let end = data.len() - 3;
    let mut i = start;
    while i < end {
        if &data[i..i + 4] == b"GRIB" {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Read a big-endian u64 from `data` at `offset`. Panics if out of bounds.
fn read_u64_be(data: &[u8], offset: usize) -> u64 {
    u64::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

// ─── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal synthetic GRIB2 message of a given total length.
    /// The message won't parse correctly (no valid sections), but it has
    /// the right magic + edition + length header for the streaming parser
    /// to recognize message boundaries.
    fn make_fake_grib2(total_len: usize) -> Vec<u8> {
        assert!(total_len >= 20, "GRIB2 message must be >= 20 bytes");
        let mut msg = vec![0u8; total_len];
        // Magic
        msg[0] = b'G';
        msg[1] = b'R';
        msg[2] = b'I';
        msg[3] = b'B';
        // Reserved
        msg[4] = 0;
        msg[5] = 0;
        // Discipline
        msg[6] = 0;
        // Edition = 2
        msg[7] = 2;
        // Total length as big-endian u64
        let len_bytes = (total_len as u64).to_be_bytes();
        msg[8..16].copy_from_slice(&len_bytes);
        // End marker "7777" at the end
        let end = total_len;
        msg[end - 4] = b'7';
        msg[end - 3] = b'7';
        msg[end - 2] = b'7';
        msg[end - 1] = b'7';
        msg
    }

    #[test]
    fn test_streaming_single_message_one_chunk() {
        // Build a fake GRIB2 message. It won't fully parse (no valid sections),
        // but we can test the boundary detection logic by checking buffered_bytes.
        let msg = make_fake_grib2(64);

        let mut parser = StreamingParser::new();
        // Feed the whole thing at once — the parse will fail because it's not
        // a real GRIB2 message, but the boundary detection should consume all bytes.
        let _ = parser.feed(&msg);
        // Buffer should be empty after consuming the message (even if parse fails)
        assert_eq!(parser.buffered_bytes(), 0);
    }

    #[test]
    fn test_streaming_incremental_feed() {
        let msg = make_fake_grib2(64);

        let mut parser = StreamingParser::new();

        // Feed just the first 10 bytes — not enough for indicator
        parser.feed(&msg[..10]);
        assert_eq!(parser.buffered_bytes(), 10);

        // Feed up to byte 20 — indicator complete, but not full message
        parser.feed(&msg[10..20]);
        assert_eq!(parser.buffered_bytes(), 20);

        // Feed the rest — should now detect the complete message boundary
        let _ = parser.feed(&msg[20..]);
        assert_eq!(parser.buffered_bytes(), 0);
    }

    #[test]
    fn test_streaming_two_messages() {
        let msg1 = make_fake_grib2(40);
        let msg2 = make_fake_grib2(48);

        let mut combined = Vec::new();
        combined.extend_from_slice(&msg1);
        combined.extend_from_slice(&msg2);

        let mut parser = StreamingParser::new();
        let _ = parser.feed(&combined);
        // Both messages should be consumed from the buffer
        assert_eq!(parser.buffered_bytes(), 0);
    }

    #[test]
    fn test_streaming_split_across_magic() {
        let msg = make_fake_grib2(64);

        let mut parser = StreamingParser::new();

        // Feed just "GRI" — partial magic
        parser.feed(&msg[..3]);
        // Feed the rest starting from "B..."
        let _ = parser.feed(&msg[3..]);
        assert_eq!(parser.buffered_bytes(), 0);
    }

    #[test]
    fn test_finish_with_leftover_junk() {
        let mut parser = StreamingParser::new();
        // Feed some non-GRIB junk
        parser.feed(b"hello world this is not grib data");
        // finish should succeed (no partial GRIB magic found)
        let result = parser.finish();
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_finish_with_partial_message() {
        let msg = make_fake_grib2(64);

        let mut parser = StreamingParser::new();
        // Feed only part of the message
        parser.feed(&msg[..32]);
        // finish should return an error about incomplete message
        let result = parser.finish();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Incomplete"), "Error: {}", err);
    }

    #[test]
    fn test_has_messages_and_take() {
        let mut parser = StreamingParser::new();
        assert!(!parser.has_messages());
        assert_eq!(parser.pending_count(), 0);

        let taken = parser.take_messages();
        assert!(taken.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let parser = StreamingParser::with_capacity(1024 * 1024);
        assert_eq!(parser.buffered_bytes(), 0);
        assert_eq!(parser.total_parsed(), 0);
    }

    #[test]
    fn test_default_trait() {
        let parser = StreamingParser::default();
        assert_eq!(parser.buffered_bytes(), 0);
    }

    #[test]
    fn test_junk_before_magic() {
        let mut data = Vec::new();
        // 10 bytes of junk before the actual message
        data.extend_from_slice(b"JUNKJUNKJU");
        data.extend_from_slice(&make_fake_grib2(40));

        let mut parser = StreamingParser::new();
        let _ = parser.feed(&data);
        // The junk + message should both be consumed
        assert_eq!(parser.buffered_bytes(), 0);
    }

    #[test]
    fn test_edition_1_skipped() {
        // Create a message with edition=1 — should be skipped
        let mut msg = make_fake_grib2(40);
        msg[7] = 1; // edition 1

        let mut parser = StreamingParser::new();
        parser.feed(&msg);
        // Should have scanned past it (only last 3 bytes kept as potential partial magic)
        assert!(parser.buffered_bytes() <= 3);
    }
}
