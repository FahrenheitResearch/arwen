//! A streaming reader for the daily tarballs the CDAAC radio-occultation
//! archive publishes (`atmPrf_nrt_YYYY_DDD.tar.gz`: six thousand
//! occultation files a day, two gigabytes gzipped, five expanded).
//!
//! The vendor closure carries no tar crate, and the archive needs only
//! the POSIX ustar layout: 512-byte headers, an octal (or base-256) size,
//! the name in the header's first hundred bytes with the ustar prefix in
//! front of it, and GNU's `L` entry for a name longer than that.  Members
//! are yielded one at a time as `(name, bytes)` so a two-gigabyte archive
//! never has to be expanded onto a disk or into memory whole; the caller
//! decodes each member and lets it go.
//!
//! Fail-closed: a header whose checksum does not add up, a size that does
//! not parse, or a member cut short by the end of the stream is an error
//! naming the member, never a shorter file handed back as if it were whole.

use std::error::Error;
use std::io::Read;

use crate::err;

const BLOCK: usize = 512;

/// One regular file of a tar stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TarMember {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// The regular files of a tar stream, in order.
pub struct TarMembers<R: Read> {
    reader: R,
    subject: String,
    finished: bool,
    /// A GNU long name (`L` entry) waiting for the entry it names.
    pending_long_name: Option<String>,
    /// Members whose size field could not be read as a number: skipped
    /// is not an option (their length is unknown), so the stream ends.
    pub members_seen: usize,
}

impl<R: Read> TarMembers<R> {
    pub fn new(reader: R, subject: impl Into<String>) -> Self {
        Self { reader, subject: subject.into(), finished: false, pending_long_name: None, members_seen: 0 }
    }

    fn read_exact_or_none(&mut self, buffer: &mut [u8]) -> Result<bool, Box<dyn Error>> {
        let mut filled = 0;
        while filled < buffer.len() {
            let n = self
                .reader
                .read(&mut buffer[filled..])
                .map_err(|e| err(format!("{}: reading the tar stream: {e}", self.subject)))?;
            if n == 0 {
                if filled == 0 {
                    return Ok(false);
                }
                return Err(err(format!(
                    "{}: the tar stream ended inside a block ({filled} of {} bytes)",
                    self.subject,
                    buffer.len()
                )));
            }
            filled += n;
        }
        Ok(true)
    }

    fn next_member(&mut self) -> Result<Option<TarMember>, Box<dyn Error>> {
        loop {
            let mut header = [0u8; BLOCK];
            if !self.read_exact_or_none(&mut header)? {
                return Ok(None);
            }
            if header.iter().all(|&b| b == 0) {
                // The end-of-archive marker (two zero blocks); the second
                // is read on the next call and answers with none as well.
                self.finished = true;
                return Ok(None);
            }
            verify_checksum(&header, &self.subject)?;
            let size = parse_size(&header[124..136], &self.subject)?;
            let typeflag = header[156];
            let name = match self.pending_long_name.take() {
                Some(long) => long,
                None => entry_name(&header),
            };
            let padded = size.div_ceil(BLOCK as u64).checked_mul(BLOCK as u64).ok_or_else(|| {
                err(format!("{}: member {name:?} declares {size} bytes, more than a tar stream can hold", self.subject))
            })?;
            // The buffer grows with the bytes the stream actually delivers,
            // never with the size a header claims: a corrupt or hostile
            // header naming terabytes ends as "cut short" below instead of
            // an allocation that aborts the process.
            let mut bytes = Vec::with_capacity(padded.min(1 << 20) as usize);
            let delivered = (&mut self.reader)
                .take(padded)
                .read_to_end(&mut bytes)
                .map_err(|e| err(format!("{}: reading the tar stream: {e}", self.subject)))?;
            if (delivered as u64) < padded {
                return Err(err(format!("{}: member {name:?} is cut short by the end of the stream", self.subject)));
            }
            // size <= padded == delivered, so it fits the buffer's usize.
            bytes.truncate(size as usize);
            self.members_seen += 1;
            match typeflag {
                b'L' => {
                    // GNU long name: the content is the next entry's name.
                    let text = String::from_utf8_lossy(&bytes).trim_end_matches('\0').to_string();
                    self.pending_long_name = Some(text);
                    continue;
                }
                b'0' | 0 | b'7' => return Ok(Some(TarMember { name, bytes })),
                // Directories, links, pax headers and the rest carry no
                // observation; skipped, their content consumed above.
                _ => continue,
            }
        }
    }
}

impl<R: Read> Iterator for TarMembers<R> {
    type Item = Result<TarMember, Box<dyn Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.next_member() {
            Ok(Some(member)) => Some(Ok(member)),
            Ok(None) => None,
            Err(e) => {
                self.finished = true;
                Some(Err(e))
            }
        }
    }
}

fn octal_field(field: &[u8]) -> Option<u64> {
    let text: String = field
        .iter()
        .take_while(|&&b| b != 0 && b != b' ')
        .map(|&b| b as char)
        .collect();
    let text = text.trim();
    if text.is_empty() {
        return Some(0);
    }
    u64::from_str_radix(text, 8).ok()
}

fn parse_size(field: &[u8], subject: &str) -> Result<u64, Box<dyn Error>> {
    if field[0] & 0x80 != 0 {
        // GNU base-256 for sizes past eight gigabytes.
        let mut value: u64 = 0;
        for (i, &b) in field.iter().enumerate() {
            let byte = if i == 0 { b & 0x7f } else { b };
            // checked_shl only refuses a shift of 64 or more bits, so it let
            // the high byte fall off and a size past 2^64 wrapped to a
            // small one; the multiply is what notices the lost bits.
            value = value
                .checked_mul(256)
                .and_then(|v| v.checked_add(u64::from(byte)))
                .ok_or_else(|| err(format!("{subject}: a base-256 size overflows")))?;
        }
        return Ok(value);
    }
    octal_field(field).ok_or_else(|| err(format!("{subject}: a tar header's size field is not octal")))
}

fn verify_checksum(header: &[u8; BLOCK], subject: &str) -> Result<(), Box<dyn Error>> {
    let stated = octal_field(&header[148..156])
        .ok_or_else(|| err(format!("{subject}: a tar header's checksum field is not octal")))?;
    let mut unsigned: u64 = 0;
    let mut signed: i64 = 0;
    for (i, &b) in header.iter().enumerate() {
        let v = if (148..156).contains(&i) { b' ' } else { b };
        unsigned += u64::from(v);
        signed += i64::from(v as i8);
    }
    if stated == unsigned || stated as i64 == signed {
        Ok(())
    } else {
        Err(err(format!(
            "{subject}: a tar header's checksum does not add up (stated {stated}, computed {unsigned}); \
             the stream is not a tar archive or is corrupt"
        )))
    }
}

fn entry_name(header: &[u8; BLOCK]) -> String {
    let field = |range: std::ops::Range<usize>| -> String {
        String::from_utf8_lossy(&header[range]).trim_end_matches('\0').to_string()
    };
    let name = field(0..100);
    let ustar = &header[257..262] == b"ustar";
    if ustar {
        let prefix = field(345..500);
        if !prefix.is_empty() {
            return format!("{prefix}/{name}");
        }
    }
    name
}

/// The members of a gzip-wrapped tar stream (`.tar.gz`); a plain tar
/// stream is read as it is.
pub fn tar_members<'a>(
    reader: impl Read + 'a,
    gzipped: bool,
    subject: impl Into<String>,
) -> Box<dyn Iterator<Item = Result<TarMember, Box<dyn Error>>> + 'a> {
    let subject = subject.into();
    if gzipped {
        Box::new(TarMembers::new(flate2::read::MultiGzDecoder::new(reader), subject))
    } else {
        Box::new(TarMembers::new(reader, subject))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A ustar entry as GNU tar writes it (for the tests; the crate reads,
    /// it does not write archives).
    pub fn ustar_entry(name: &str, bytes: &[u8], typeflag: u8) -> Vec<u8> {
        let mut header = [0u8; BLOCK];
        header[..name.len()].copy_from_slice(name.as_bytes());
        header[100..108].copy_from_slice(b"0000644\0");
        header[108..116].copy_from_slice(b"0001750\0");
        header[116..124].copy_from_slice(b"0001750\0");
        let size = format!("{:011o}\0", bytes.len());
        header[124..136].copy_from_slice(size.as_bytes());
        header[136..148].copy_from_slice(b"14700000000\0");
        header[156] = typeflag;
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        let sum: u64 = header.iter().enumerate().map(|(i, &b)| if (148..156).contains(&i) { 32 } else { u64::from(b) }).sum();
        let checksum = format!("{sum:06o}\0 ");
        header[148..156].copy_from_slice(checksum.as_bytes());
        let mut out = header.to_vec();
        out.extend_from_slice(bytes);
        let pad = (BLOCK - bytes.len() % BLOCK) % BLOCK;
        out.extend(std::iter::repeat_n(0u8, pad));
        out
    }

    pub fn archive(entries: &[(&str, &[u8], u8)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, bytes, flag) in entries {
            out.extend(ustar_entry(name, bytes, *flag));
        }
        out.extend(std::iter::repeat_n(0u8, 2 * BLOCK));
        out
    }

    #[test]
    fn regular_files_come_back_in_order_with_their_bytes_and_directories_are_skipped() {
        let tar = archive(&[
            ("atmPrf_a_nc", b"hello", b'0'),
            ("dir/", b"", b'5'),
            ("atmPrf_b_nc", &[7u8; 1000], b'0'),
        ]);
        let members: Vec<TarMember> = TarMembers::new(tar.as_slice(), "test").map(|m| m.unwrap()).collect();
        assert_eq!(members.len(), 2);
        assert_eq!(members[0].name, "atmPrf_a_nc");
        assert_eq!(members[0].bytes, b"hello");
        assert_eq!(members[1].name, "atmPrf_b_nc");
        assert_eq!(members[1].bytes.len(), 1000);
    }

    #[test]
    fn a_gnu_long_name_names_the_entry_after_it() {
        let long = "x".repeat(150);
        let tar = archive(&[("././@LongLink", long.as_bytes(), b'L'), ("truncated", b"z", b'0')]);
        let members: Vec<TarMember> = TarMembers::new(tar.as_slice(), "test").map(|m| m.unwrap()).collect();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].name, long);
    }

    #[test]
    fn a_gzip_wrapped_archive_streams_through_and_a_cut_stream_is_refused() {
        use std::io::Write;
        let tar = archive(&[("a_nc", &[1u8; 700], b'0')]);
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(&tar).unwrap();
        let gz = encoder.finish().unwrap();
        let members: Vec<TarMember> = tar_members(gz.as_slice(), true, "gz").map(|m| m.unwrap()).collect();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].bytes.len(), 700);
        let cut = &tar[..BLOCK + 300];
        let outcome: Vec<_> = TarMembers::new(cut, "cut").collect();
        assert!(outcome.len() == 1 && outcome[0].is_err());
        let message = outcome[0].as_ref().unwrap_err().to_string();
        assert!(message.contains("cut") || message.contains("ended inside"), "{message}");
    }

    /// One header whose size field is `field`, with its checksum redone.
    fn entry_with_size_field(name: &str, content: &[u8], field: &[u8; 12]) -> Vec<u8> {
        let mut bytes = ustar_entry(name, content, b'0');
        bytes[124..136].copy_from_slice(field);
        let sum: u64 = bytes[..BLOCK]
            .iter()
            .enumerate()
            .map(|(i, &b)| if (148..156).contains(&i) { 32 } else { u64::from(b) })
            .sum();
        bytes[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        bytes
    }

    fn base256(value: u128) -> [u8; 12] {
        let mut field = [0u8; 12];
        for (i, byte) in field.iter_mut().enumerate().skip(1) {
            *byte = (value >> (8 * (11 - i))) as u8;
        }
        field[0] |= 0x80;
        field
    }

    #[test]
    fn a_base256_size_past_u64_is_refused_instead_of_wrapping() {
        // 2^64 + 1 bytes used to come back as 1: the high byte was shifted
        // out and the member was handed back one byte long.
        let wrapped = base256((1u128 << 64) + 1);
        let message = parse_size(&wrapped, "size").unwrap_err().to_string();
        assert!(message.contains("overflows"), "{message}");
        let tar = entry_with_size_field("profile.nc", b"x", &wrapped);
        let outcome: Vec<_> = TarMembers::new(tar.as_slice(), "wrap").collect();
        assert!(outcome.len() == 1 && outcome[0].is_err(), "{outcome:?}");
        // The largest size a u64 holds still parses; the value is exact.
        assert_eq!(parse_size(&base256(u128::from(u64::MAX)), "size").unwrap(), u64::MAX);
        assert_eq!(parse_size(&base256(9 << 30), "size").unwrap(), 9 << 30);
    }

    #[test]
    fn a_size_that_cannot_be_padded_to_a_block_is_refused() {
        // u64::MAX rounded up to a whole block overflowed: a debug build
        // panicked and a release build wrapped the padding to zero bytes.
        let tar = entry_with_size_field("huge.nc", b"", &base256(u128::from(u64::MAX)));
        let message = TarMembers::new(tar.as_slice(), "max").next().unwrap().unwrap_err().to_string();
        assert!(message.contains("more than a tar stream can hold"), "{message}");
    }

    #[test]
    fn a_claimed_size_the_stream_does_not_hold_is_cut_short_not_allocated() {
        // A terabyte used to be allocated whole before a byte of it was
        // read, which aborts the process on any machine without it.
        let tar = entry_with_size_field("huge.nc", &[5u8; 700], &base256(1 << 40));
        let message = TarMembers::new(tar.as_slice(), "tib").next().unwrap().unwrap_err().to_string();
        assert!(message.contains("cut short"), "{message}");
    }

    #[test]
    fn a_corrupt_header_is_refused_by_its_checksum() {
        let mut tar = archive(&[("a_nc", b"abc", b'0')]);
        tar[10] ^= 0x55;
        let outcome: Vec<_> = TarMembers::new(tar.as_slice(), "bad").collect();
        assert!(outcome[0].is_err());
        assert!(outcome[0].as_ref().unwrap_err().to_string().contains("checksum"));
    }
}
