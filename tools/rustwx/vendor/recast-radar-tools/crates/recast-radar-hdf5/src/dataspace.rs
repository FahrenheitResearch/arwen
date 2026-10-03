//! Dataspace messages (HDF5 File Format Specification, section IV.A.2.b).

use crate::bytes::Cursor;
use crate::error::{Result, invalid, limit};
use crate::limits::{MAX_DATASPACE_DIM, MAX_DATASPACE_RANK};

/// The `max_dims` value of an unlimited dimension.
pub const UNLIMITED: u64 = u64::MAX;

/// Shape of a dataset or attribute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Dataspace {
    /// Null dataspace: no elements at all.
    pub(crate) null: bool,
    /// Current dimensions (empty for a scalar).
    pub(crate) dims: Vec<u64>,
    /// Maximum dimensions ([`UNLIMITED`] = unlimited), when stored.
    pub(crate) max_dims: Option<Vec<u64>>,
}

impl Dataspace {
    /// Number of elements: 1 for a scalar, 0 for a null dataspace.
    pub(crate) fn element_count(&self) -> Result<usize> {
        if self.null {
            return Ok(0);
        }
        self.dims.iter().try_fold(1usize, |product, dim| {
            usize::try_from(*dim)
                .ok()
                .and_then(|dim| product.checked_mul(dim))
                .ok_or_else(|| invalid(0, "HDF5 dataspace element count overflow"))
        })
    }

    pub(crate) fn dims_usize(&self) -> Vec<usize> {
        // Dimensions were checked against MAX_DATASPACE_DIM at parse time.
        self.dims.iter().map(|dim| *dim as usize).collect()
    }
}

/// Version 1: version, rank, flags, reserved (5), dimensions, [maximum
/// dimensions when flags bit 0], [permutation indices when flags bit 1].
/// Version 2: version, rank, flags, type (0 scalar, 1 simple, 2 null),
/// dimensions, [maximum dimensions]. Sizes are length-size fields.
pub(crate) fn parse(body: &[u8], offset: usize, length_size: usize) -> Result<Dataspace> {
    let mut cursor = Cursor::new(body, offset);
    let version = cursor.u8()?;
    let rank = usize::from(cursor.u8()?);
    let flags = cursor.u8()?;
    if rank > MAX_DATASPACE_RANK {
        return Err(limit(format!(
            "HDF5 dataspace rank is {rank} (limit {MAX_DATASPACE_RANK})"
        )));
    }
    let null = match version {
        1 => {
            cursor.skip(5)?;
            false
        }
        2 => {
            let kind = cursor.u8()?;
            match kind {
                0 | 1 => false,
                2 => true,
                other => {
                    return Err(invalid(offset, format!("dataspace type {other} unknown")));
                }
            }
        }
        other => {
            return Err(invalid(
                offset,
                format!("dataspace version {other} unsupported"),
            ));
        }
    };
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        let dim = cursor.uint(length_size)?;
        if dim > MAX_DATASPACE_DIM {
            return Err(limit(format!(
                "HDF5 dataspace dimension is {dim} (limit {MAX_DATASPACE_DIM})"
            )));
        }
        dims.push(dim);
    }
    let max_dims = if flags & 0x01 != 0 {
        let mut max_dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            let dim = cursor.uint(length_size)?;
            let all_ones = length_size < 8 && dim == (1u64 << (8 * length_size)) - 1;
            max_dims.push(if all_ones { UNLIMITED } else { dim });
        }
        Some(max_dims)
    } else {
        None
    };
    Ok(Dataspace {
        null,
        dims,
        max_dims,
    })
}
