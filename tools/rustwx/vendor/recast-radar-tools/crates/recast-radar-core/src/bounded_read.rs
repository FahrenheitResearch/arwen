//! Resource limits shared by the format decoders and the format router.
//!
//! Every decoder bounds the memory it allocates from header values, so a
//! small or corrupt file cannot make it allocate far more than its own bytes
//! justify. The shared ceilings live here; format-specific ones (HDF5 B-tree
//! nodes, netCDF attribute counts, GRIB2 sections, ...) live in each decoder
//! crate and are listed in that crate's `# Limits` documentation.
//!
//! | Limit | Value | Largest real need in the test corpus |
//! |---|---|---|
//! | [`MAX_DECODED_RADAR_BYTES`] | 512 MiB | 83.5 MiB (decompressed Level II `PGUA` volume) |
//! | [`MAX_DECODED_VOLUME_BYTES`] | 1 GiB | 80 MiB (decoded Level II `PGUA` volume) |
//! | [`MAX_DECODED_BATCH_BYTES`] | 2 GiB | 586 MiB (all 20 stations of the JMA N5 tar) |
//! | [`MAX_GATES_PER_RADIAL`] | 16,384 | 1,840 (Level II) |
//! | [`MAX_SWEEPS_PER_VOLUME`] | 1,024 | 26 (one JMA station), 23 (Level II) |
//!
//! Decoded output is accounted with a [`DecodeBudget`]: a decoder checks the
//! growth an operation needs before allocating and records the capacity it
//! actually allocated afterwards. A decode that would exceed its budget
//! returns an error. Because a growing buffer is recorded after it
//! reallocates, a failing decode can briefly pass the ceiling by one
//! reallocation of its largest field.
//!
//! Errors are returned as the complete diagnostic message; each decoder
//! crate wraps it in its own error variant.

use std::io::Read;

use crate::model::{Field, FieldData, Volume};

/// Hard ceiling for one expanded radar payload. Operational Level II,
/// ODIM, CfRadial, and DORADE files are far smaller; this remains generous
/// enough for unusually dense research volumes while bounding compression
/// bombs before they can exhaust the process address space.
pub const MAX_DECODED_RADAR_BYTES: usize = 512 * 1024 * 1024;

/// Ceiling on the output one decoded volume retains: the allocated bytes of
/// every field's value buffer and the per-ray tables
/// a decoder charges to its [`DecodeBudget`].
pub const MAX_DECODED_VOLUME_BYTES: usize = 1024 * 1024 * 1024;

/// Ceiling on the combined output of one call that decodes several volumes
/// (every station of a JMA tar, every scan of a mobile-radar archive).
pub const MAX_DECODED_BATCH_BYTES: usize = 2 * MAX_DECODED_VOLUME_BYTES;

/// Most gates (range bins) accepted on one radial. Real maxima: 1,840
/// (NEXRAD Level II), 1,107 (CfRadial), 1,002 (DORADE), 960 (ODIM_H5).
pub const MAX_GATES_PER_RADIAL: usize = 16 * 1024;

/// Most sweeps (elevation cuts) accepted in one volume. Real maxima: 26
/// (one JMA station), 23 (NEXRAD Level II), 11 (ODIM_H5).
pub const MAX_SWEEPS_PER_VOLUME: usize = 1024;

/// Running total of decoded output bytes checked against a fixed ceiling.
///
/// Decoders [`check`](Self::check) the growth an allocation needs before
/// making it, [`charge`](Self::charge) tables of known size, and
/// [`update`](Self::update) the total with a buffer's allocated size after it
/// changes. Every error is the complete diagnostic message and names the
/// limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeBudget {
    used: usize,
    limit: usize,
}

impl DecodeBudget {
    /// An empty budget with the given ceiling in bytes.
    pub const fn new(limit: usize) -> Self {
        Self { used: 0, limit }
    }

    /// An empty budget for one volume ([`MAX_DECODED_VOLUME_BYTES`]).
    pub const fn volume() -> Self {
        Self::new(MAX_DECODED_VOLUME_BYTES)
    }

    /// Bytes committed so far.
    pub const fn used(&self) -> usize {
        self.used
    }

    /// The ceiling in bytes.
    pub const fn limit(&self) -> usize {
        self.limit
    }

    /// Bytes still available.
    pub const fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.used)
    }

    /// Succeed when `bytes` more would fit, without committing them.
    pub fn check(&self, bytes: usize, context: &str) -> Result<(), String> {
        if bytes <= self.remaining() {
            Ok(())
        } else {
            Err(format!(
                "{context} needs {bytes} more decoded bytes with {} of the {}-byte limit already used",
                self.used, self.limit
            ))
        }
    }

    /// Commit `count` elements of `element_bytes` bytes each.
    pub fn charge(
        &mut self,
        count: usize,
        element_bytes: usize,
        context: &str,
    ) -> Result<(), String> {
        let bytes = count
            .checked_mul(element_bytes)
            .ok_or_else(|| format!("{context}: decoded size overflows the address space"))?;
        self.check(bytes, context)?;
        self.used += bytes;
        Ok(())
    }

    /// Replace a previously committed buffer size `old` with its current
    /// size `new` after the buffer grew or shrank. Errors when the total now
    /// exceeds the ceiling; the new size stays committed either way.
    pub fn update(&mut self, old: usize, new: usize, context: &str) -> Result<(), String> {
        self.used = self.used.saturating_sub(old).saturating_add(new);
        if self.used <= self.limit {
            Ok(())
        } else {
            Err(format!(
                "{context} grew decoded output to {} bytes (limit {})",
                self.used, self.limit
            ))
        }
    }

    /// Give back `bytes` previously committed (a transient buffer was freed).
    pub fn release(&mut self, bytes: usize) {
        self.used = self.used.saturating_sub(bytes);
    }
}

/// Bytes allocated by a field's value buffer.
pub fn field_capacity_bytes(field: &Field) -> usize {
    match &field.data {
        FieldData::U8 { values, .. } => values.capacity(),
        FieldData::I8 { values, .. } => values.capacity(),
        FieldData::U16 { values, .. } => values.capacity().saturating_mul(2),
        FieldData::I16 { values, .. } => values.capacity().saturating_mul(2),
        FieldData::I32 { values, .. } => values.capacity().saturating_mul(4),
        FieldData::F32 { values, .. } => values.capacity().saturating_mul(4),
        FieldData::F64 { values, .. } => values.capacity().saturating_mul(8),
    }
}

/// Allocated bytes of every field's value buffer in a volume.
pub fn volume_field_capacity_bytes(volume: &Volume) -> usize {
    volume
        .sweeps
        .iter()
        .flat_map(|sweep| sweep.fields.iter())
        .fold(0usize, |total, field| {
            total.saturating_add(field_capacity_bytes(field))
        })
}

/// Succeed when `gates` is within [`MAX_GATES_PER_RADIAL`].
pub fn check_gate_count(gates: usize, context: &str) -> Result<(), String> {
    if gates <= MAX_GATES_PER_RADIAL {
        Ok(())
    } else {
        Err(format!(
            "{context} declares {gates} gates per radial (limit {MAX_GATES_PER_RADIAL})"
        ))
    }
}

/// Succeed when `sweeps` is within [`MAX_SWEEPS_PER_VOLUME`].
pub fn check_sweep_count(sweeps: usize, context: &str) -> Result<(), String> {
    if sweeps <= MAX_SWEEPS_PER_VOLUME {
        Ok(())
    } else {
        Err(format!(
            "{context} declares {sweeps} sweeps (limit {MAX_SWEEPS_PER_VOLUME})"
        ))
    }
}

/// Read an expanded stream without ever growing the destination beyond
/// `limit`. `try_reserve` turns address-space pressure into a normal decode
/// error instead of invoking the infallible allocation path.
pub fn read_to_end_limited(
    mut reader: impl Read,
    limit: usize,
    context: &'static str,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let remaining = limit.saturating_sub(output.len());
        if remaining == 0 {
            let mut probe = [0u8; 1];
            let count = reader
                .read(&mut probe)
                .map_err(|err| format!("{context}: {err}"))?;
            if count != 0 {
                return Err(format!("{context} expands beyond the {limit}-byte limit"));
            }
            break;
        }
        let read_len = remaining.min(chunk.len());
        let count = reader
            .read(&mut chunk[..read_len])
            .map_err(|err| format!("{context}: {err}"))?;
        if count == 0 {
            break;
        }
        output
            .try_reserve(count)
            .map_err(|err| format!("{context}: cannot reserve decoded buffer: {err}"))?;
        output.extend_from_slice(&chunk[..count]);
    }
    Ok(output)
}

/// Copy `bytes` into a new buffer, rejecting inputs longer than `limit` and
/// reporting allocation failure as an error.
pub fn copy_bytes_limited(
    bytes: &[u8],
    limit: usize,
    context: &'static str,
) -> Result<Vec<u8>, String> {
    if bytes.len() > limit {
        return Err(format!(
            "{context} is {} bytes (limit {limit})",
            bytes.len()
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes.len())
        .map_err(|err| format!("{context}: cannot reserve decoded buffer: {err}"))?;
    output.extend_from_slice(bytes);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_checks_before_committing_and_names_the_limit() {
        let mut budget = DecodeBudget::new(100);
        budget.charge(10, 4, "rows").expect("40 bytes fit");
        assert_eq!(budget.used(), 40);
        assert_eq!(budget.remaining(), 60);
        let error = budget
            .charge(61, 1, "rows")
            .expect_err("101 bytes do not fit");
        assert!(error.contains("limit"), "{error}");
        assert_eq!(budget.used(), 40, "a rejected charge commits nothing");
        assert!(budget.charge(usize::MAX, 2, "rows").is_err());
        budget
            .check(60, "grid")
            .expect("exactly the remainder fits");
        budget.release(30);
        assert_eq!(budget.used(), 10);
    }

    #[test]
    fn budget_update_tracks_growth_and_reports_overshoot() {
        let mut budget = DecodeBudget::new(100);
        budget.charge(1, 50, "grid").expect("fits");
        budget
            .update(50, 90, "grid")
            .expect("grew within the limit");
        assert_eq!(budget.used(), 90);
        let error = budget.update(90, 120, "grid").expect_err("over the limit");
        assert!(error.contains("limit"), "{error}");
        assert_eq!(budget.used(), 120);
        budget.update(120, 10, "grid").expect("shrank back");
        assert_eq!(budget.used(), 10);
    }

    #[test]
    fn gate_and_sweep_checks_accept_the_ceiling_and_reject_beyond() {
        check_gate_count(MAX_GATES_PER_RADIAL, "row").expect("ceiling accepted");
        let error = check_gate_count(MAX_GATES_PER_RADIAL + 1, "row").expect_err("over");
        assert!(error.contains("limit"), "{error}");
        check_sweep_count(MAX_SWEEPS_PER_VOLUME, "volume").expect("ceiling accepted");
        assert!(check_sweep_count(MAX_SWEEPS_PER_VOLUME + 1, "volume").is_err());
    }
}
