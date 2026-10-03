//! The writer's compression seam: every bzip2 stream and the gzip wrapper
//! are produced here and nowhere else.
//!
//! bzip2 streams come from `recast-radar-bzip2`'s encoder at level 9 (the
//! 900k block size of NOAA's LDM records), which writes libbzip2 1.0.8's
//! stream for the same input byte for byte except in one field of a block
//! that repeats a shorter string (its crate documentation, *Compressing*).
//! Gzip uses `flate2` with the zlib-rs backend.

use std::io::Write;

use recast_radar_bzip2::{EncoderPool, Level};

use super::WriteError;

/// Compresses LDM records: a 4-byte big-endian control word holding the
/// compressed length, then one bzip2 stream per record.
///
/// Records are compressed in parallel on the current rayon pool. The
/// encoders, and their work buffers (about 20 MB each at level 9), are
/// reused for every record while this value lives: at most one per thread
/// that has compressed a record for it. Keep one for a volume, or for a
/// real-time chunk writer's volume, and drop it afterwards to free them.
#[derive(Debug)]
pub(crate) struct LdmCompressor {
    pool: EncoderPool,
}

impl LdmCompressor {
    /// A compressor writing NOAA's block size (level 9).
    pub(crate) fn new() -> Self {
        Self {
            pool: EncoderPool::new(Level::BEST),
        }
    }

    /// Compress each record into an LDM record. The control word of the
    /// batch's last record is negated when `ends_file` (the file's last
    /// record, as NOAA's files have it).
    pub(crate) fn batch(
        &self,
        records: &[Vec<u8>],
        ends_file: bool,
    ) -> Result<Vec<Vec<u8>>, WriteError> {
        let streams = self.pool.encode_many(records);
        let last = streams.len().saturating_sub(1);
        streams
            .into_iter()
            .enumerate()
            .map(|(index, stream)| ldm_frame(stream, ends_file && index == last))
            .collect()
    }

    /// One LDM record: the control word (the compressed length, negated
    /// when `last`), then the bzip2 stream of `record`.
    pub(crate) fn record(&self, record: &[u8], last: bool) -> Result<Vec<u8>, WriteError> {
        let stream = self
            .pool
            .encode_many(&[record])
            .pop()
            .ok_or_else(|| WriteError::Compression("bzip2: no stream for the record".into()))?;
        ldm_frame(stream, last)
    }
}

/// The control word and `stream`.
fn ldm_frame(stream: Vec<u8>, last: bool) -> Result<Vec<u8>, WriteError> {
    let len = i32::try_from(stream.len()).map_err(|_| {
        WriteError::LimitExceeded(format!(
            "compressed record of {} bytes does not fit its control word",
            stream.len()
        ))
    })?;
    let control = if last { -len } else { len };
    let mut out = Vec::new();
    out.try_reserve_exact(4 + stream.len())
        .map_err(|err| WriteError::LimitExceeded(format!("compressed record: {err}")))?;
    out.extend_from_slice(&control.to_be_bytes());
    out.extend_from_slice(&stream);
    Ok(out)
}

/// A writer that wraps everything written to it in one gzip member on
/// `out` (default level); `finish` ends the member.
pub(crate) fn gzip_writer<W: Write>(out: W) -> flate2::write::GzEncoder<W> {
    flate2::write::GzEncoder::new(out, flate2::Compression::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real Level II record (the metadata record of the committed KTLX
    /// 2024 trim, decompressed), compressed here: the stream decodes back to
    /// the record with the `bzip2` crate (libbz2-rs-sys, a port of libbzip2
    /// 1.0.8) and equals the stream that crate writes at level 9.
    #[test]
    fn records_are_libbzip2_streams() {
        use std::io::Read;

        let raw = recast_radar_testdata::bytes("l2-ktlx-20240315-000217-trim").unwrap();
        // Volume header, then the first LDM record: control word, stream.
        let control = i32::from_be_bytes([raw[24], raw[25], raw[26], raw[27]]);
        let stream = &raw[28..28 + control.unsigned_abs() as usize];
        let mut record = Vec::new();
        bzip2::read::BzDecoder::new(stream)
            .read_to_end(&mut record)
            .unwrap();
        assert!(record.len() > 100_000, "{}", record.len());

        let compressor = LdmCompressor::new();
        let ours = compressor.record(&record, false).unwrap();
        let len = i32::from_be_bytes([ours[0], ours[1], ours[2], ours[3]]);
        assert_eq!(len as usize, ours.len() - 4);
        let mut back = Vec::new();
        bzip2::read::BzDecoder::new(&ours[4..])
            .read_to_end(&mut back)
            .unwrap();
        assert_eq!(back, record);

        let mut reference = Vec::new();
        bzip2::write::BzEncoder::new(&mut reference, bzip2::Compression::best())
            .write_all(&record)
            .unwrap();
        // `BzEncoder` finishes the stream when dropped.
        assert_eq!(&ours[4..], &reference[..]);

        // A batch negates the last control word only when it ends the file.
        let batch = compressor
            .batch(&[record.clone(), record.clone()], true)
            .unwrap();
        assert!(i32::from_be_bytes([batch[0][0], batch[0][1], batch[0][2], batch[0][3]]) > 0);
        assert!(i32::from_be_bytes([batch[1][0], batch[1][1], batch[1][2], batch[1][3]]) < 0);
        assert_eq!(batch[0], ours);
    }
}
