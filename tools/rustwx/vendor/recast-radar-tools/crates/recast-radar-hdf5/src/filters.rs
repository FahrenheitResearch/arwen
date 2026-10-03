//! Filter pipeline messages (section IV.A.2.l) and the inverse filters:
//! deflate, shuffle and Fletcher-32. Other filters are typed
//! [`Error::UnsupportedFilter`] errors.

use flate2::{Decompress, FlushDecompress, Status};

use crate::bytes::Cursor;
use crate::checksum::fletcher32;
use crate::error::{Error, Result, invalid, limit};
use crate::limits::{MAX_DATASET_BYTES, MAX_FILTER_VALUES, MAX_FILTERS};

/// HDF5 filter identifiers this crate knows by name.
const FILTER_DEFLATE: u16 = 1;
const FILTER_SHUFFLE: u16 = 2;
const FILTER_FLETCHER32: u16 = 3;

/// One filter of a dataset's (or fractal heap's) pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Filter {
    /// Registered filter identifier (1 deflate, 2 shuffle, 3 Fletcher-32,
    /// 4 szip, 5 n-bit, 6 scale-offset, 32000 LZF, ...).
    pub id: u16,
    /// Name stored in the file, or the registered name.
    pub name: String,
    /// The filter may fail on write without failing the write (flags bit 0).
    pub optional: bool,
    /// Client data values.
    pub client_values: Vec<u32>,
}

/// Registered name of a filter id.
fn registered_name(id: u16) -> &'static str {
    match id {
        1 => "deflate",
        2 => "shuffle",
        3 => "fletcher32",
        4 => "szip",
        5 => "nbit",
        6 => "scaleoffset",
        307 => "bzip2",
        32000 => "lzf",
        32001 => "blosc",
        32004 => "lz4",
        32008 => "bitshuffle",
        32013 => "zfp",
        32015 => "zstd",
        _ => "unregistered",
    }
}

/// Version 1: version, filter count, 6 reserved bytes, then per filter id
/// (u16), name length (u16), flags (u16), client value count (u16), name
/// padded to 8 bytes, values (u32 each), 4 padding bytes after an odd count.
/// Version 2: version, count, then id, [name length when id >= 256], flags,
/// value count, [unpadded name], values.
pub(crate) fn parse(body: &[u8], offset: usize) -> Result<Vec<Filter>> {
    let mut cursor = Cursor::new(body, offset);
    let version = cursor.u8()?;
    let count = usize::from(cursor.u8()?);
    if count > MAX_FILTERS {
        return Err(limit(format!(
            "HDF5 filter pipeline has {count} filters (limit {MAX_FILTERS})"
        )));
    }
    match version {
        1 => cursor.skip(6)?,
        2 => {}
        other => {
            return Err(invalid(
                offset,
                format!("filter pipeline version {other} unsupported"),
            ));
        }
    }
    let mut filters = Vec::with_capacity(count);
    for _ in 0..count {
        let id = cursor.u16()?;
        let has_name = version == 1 || id >= 256;
        let name_len = if has_name {
            usize::from(cursor.u16()?)
        } else {
            0
        };
        let flags = cursor.u16()?;
        let value_count = usize::from(cursor.u16()?);
        if value_count > MAX_FILTER_VALUES {
            return Err(limit(format!(
                "HDF5 filter has {value_count} client values (limit {MAX_FILTER_VALUES})"
            )));
        }
        let name = if name_len > 0 {
            let padded = if version == 1 {
                name_len.div_ceil(8) * 8
            } else {
                name_len
            };
            let raw = cursor.take(padded)?;
            let raw = raw.split(|byte| *byte == 0).next().unwrap_or_default();
            String::from_utf8_lossy(raw).into_owned()
        } else {
            registered_name(id).to_owned()
        };
        let mut client_values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            client_values.push(cursor.u32()?);
        }
        if version == 1 && value_count % 2 == 1 {
            cursor.skip(4)?;
        }
        filters.push(Filter {
            id,
            name,
            optional: flags & 0x01 != 0,
            client_values,
        });
    }
    Ok(filters)
}

/// Reject a pipeline this crate cannot invert, before any chunk is read.
pub(crate) fn check_supported(filters: &[Filter]) -> Result<()> {
    for filter in filters {
        if !matches!(
            filter.id,
            FILTER_DEFLATE | FILTER_SHUFFLE | FILTER_FLETCHER32
        ) {
            return Err(Error::UnsupportedFilter {
                id: filter.id,
                name: filter.name.clone(),
            });
        }
    }
    Ok(())
}

/// Run the inverse pipeline over one stored block. Filters undo in reverse
/// order; bit N of `mask` set means filter N was skipped on write. The
/// result holds at most `expected + 64` bytes (the decoded chunk plus room
/// for checksums a pipeline appended before compressing).
pub(crate) fn apply_inverse(
    stored: &[u8],
    filters: &[Filter],
    mask: u32,
    element_size: usize,
    expected: usize,
) -> Result<Vec<u8>> {
    let max_output = expected.saturating_add(64);
    if stored.len() > MAX_DATASET_BYTES {
        return Err(limit(format!(
            "HDF5 stored filter input is {} bytes (limit {MAX_DATASET_BYTES})",
            stored.len()
        )));
    }
    let mut data: Option<Vec<u8>> = None;
    for (index, filter) in filters.iter().enumerate().rev() {
        if index < 32 && mask & (1 << index) != 0 {
            continue;
        }
        let input: &[u8] = data.as_deref().unwrap_or(stored);
        let output = match filter.id {
            FILTER_DEFLATE => inflate(input, max_output)?,
            FILTER_SHUFFLE => {
                let size = filter
                    .client_values
                    .first()
                    .map_or(element_size, |value| *value as usize)
                    .max(1);
                unshuffle(input, size)
            }
            FILTER_FLETCHER32 => verify_fletcher32(input)?.to_vec(),
            other => {
                return Err(Error::UnsupportedFilter {
                    id: other,
                    name: filter.name.clone(),
                });
            }
        };
        if output.len() > max_output {
            return Err(limit(format!(
                "HDF5 filter output exceeds its {expected}-byte chunk size (limit)"
            )));
        }
        data = Some(output);
    }
    Ok(data.unwrap_or_else(|| stored.to_vec()))
}

/// zlib inflate into at most `max_output` bytes.
fn inflate(input: &[u8], max_output: usize) -> Result<Vec<u8>> {
    let mut decoder = Decompress::new(true);
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(max_output.min(MAX_DATASET_BYTES + 64))
        .map_err(|err| invalid(0, format!("cannot reserve HDF5 deflate output: {err}")))?;
    loop {
        let consumed = decoder.total_in() as usize;
        let before = out.len();
        let status = decoder
            .decompress_vec(
                input.get(consumed..).unwrap_or_default(),
                &mut out,
                FlushDecompress::Finish,
            )
            .map_err(|err| invalid(0, format!("HDF5 deflate chunk: {err}")))?;
        match status {
            Status::StreamEnd => return Ok(out),
            Status::Ok | Status::BufError => {
                if out.len() == out.capacity() {
                    return Err(limit(format!(
                        "HDF5 deflate chunk expands beyond its {}-byte chunk size (limit)",
                        max_output.saturating_sub(64)
                    )));
                }
                if out.len() == before && decoder.total_in() as usize == consumed {
                    return Err(invalid(0, "HDF5 deflate chunk: truncated stream"));
                }
            }
        }
    }
}

/// The Fletcher-32 filter appends a little-endian checksum of the data; the
/// library also accepts the byte-swapped value HDF5 1.6.2 and earlier wrote.
fn verify_fletcher32(input: &[u8]) -> Result<&[u8]> {
    let Some((data, stored)) = input.split_last_chunk::<4>() else {
        return Err(invalid(
            0,
            "HDF5 Fletcher-32 chunk shorter than its checksum",
        ));
    };
    let stored = u32::from_le_bytes(*stored);
    let computed = fletcher32(data);
    let bytes = computed.to_le_bytes();
    let reversed = u32::from_le_bytes([bytes[1], bytes[0], bytes[3], bytes[2]]);
    if stored != computed && stored != reversed {
        return Err(Error::Checksum {
            structure: "HDF5 Fletcher-32 chunk",
            offset: 0,
            stored,
            computed,
        });
    }
    Ok(data)
}

/// Inverse of the shuffle filter: byte plane k holds byte k of every
/// element. Trailing bytes that do not fill an element stay in place.
pub(crate) fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    let count = data.len() / element_size.max(1);
    if element_size <= 1 || count <= 1 {
        return data.to_vec();
    }
    let mut out = vec![0u8; data.len()];
    for plane in 0..element_size {
        let source = &data[plane * count..(plane + 1) * count];
        for (element, byte) in source.iter().enumerate() {
            out[element * element_size + plane] = *byte;
        }
    }
    let tail = count * element_size;
    out[tail..].copy_from_slice(&data[tail..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshuffle_reinterleaves_byte_planes() {
        // Two u16 elements 0x0201, 0x0403 shuffled = planes [01 03][02 04].
        let shuffled = [0x01, 0x03, 0x02, 0x04];
        assert_eq!(unshuffle(&shuffled, 2), vec![0x01, 0x02, 0x03, 0x04]);
        // A trailing partial element is copied through.
        assert_eq!(unshuffle(&[1, 3, 2, 4, 9], 2), vec![1, 2, 3, 4, 9]);
    }
}
