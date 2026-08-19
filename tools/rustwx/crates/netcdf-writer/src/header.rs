//! Big-endian serialization of the classic header grammar.
//!
//! ```text
//! header   := magic numrecs dim_list gatt_list var_list
//! magic    := 'C' 'D' 'F' VERSION            VERSION = \x01 | \x02 | \x05
//! numrecs  := NON_NEG
//! dim_list := ABSENT | NC_DIMENSION(0x0A) nelems [dim ...]
//!   dim    := name  dim_length               (dim_length 0 = the record dim)
//! att_list := ABSENT | NC_ATTRIBUTE(0x0C) nelems [attr ...]
//!   attr   := name  nc_type(u32)  nelems  [values, zero-padded to 4]
//! var_list := ABSENT | NC_VARIABLE(0x0B) nelems [var ...]
//!   var    := name  nelems  [dimid ...]  vatt_list  nc_type(u32) vsize begin
//! ABSENT   := ZERO(u32) NON_NEG(0)
//! name     := nelems(byte length, NOT counting padding) bytes pad-to-4
//! ```
//!
//! `NON_NEG` is 4 bytes in CDF-1/CDF-2 and 8 bytes in CDF-5; that single
//! rule covers `numrecs`, every `nelems`, every name length, `dim_length`,
//! every `dimid` and `vsize`. `begin` is 4 bytes in CDF-1 and 8 bytes in
//! CDF-2/CDF-5, and the tags (`NC_DIMENSION`/`NC_ATTRIBUTE`/`NC_VARIABLE`
//! and the ABSENT zero) are always 4 bytes in every version.
//!
//! MEASURED against files written by netCDF-C (netCDF4-python 1.7.4,
//! `NETCDF3_64BIT_OFFSET` and `NETCDF3_64BIT_DATA`) on 2026-08-16: the
//! CDF-5 `ABSENT` encoding is a 4-byte zero tag followed by an 8-byte
//! zero count (12 bytes), and `nc_type` stays 4 bytes in CDF-5.

use crate::layout::VarLayout;
use crate::schema::{Attr, Schema};
use crate::types::NcFormat;

const NC_DIMENSION: u32 = 0x0000_000A;
const NC_VARIABLE: u32 = 0x0000_000B;
const NC_ATTRIBUTE: u32 = 0x0000_000C;
const ABSENT_TAG: u32 = 0x0000_0000;

/// Round up to the next multiple of 4.
#[inline]
pub(crate) fn pad4(n: u64) -> u64 {
    (n + 3) & !3
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

/// A `NON_NEG` count: 4 bytes, or 8 in CDF-5.
fn put_count(buf: &mut Vec<u8>, v: u64, format: NcFormat) {
    if format.wide_counts() {
        buf.extend_from_slice(&v.to_be_bytes());
    } else {
        buf.extend_from_slice(&(v as u32).to_be_bytes());
    }
}

/// An `OFFSET`: 4 bytes in CDF-1, 8 in CDF-2/CDF-5.
fn put_offset(buf: &mut Vec<u8>, v: u64, format: NcFormat) {
    if format.wide_offsets() {
        buf.extend_from_slice(&v.to_be_bytes());
    } else {
        buf.extend_from_slice(&(v as u32).to_be_bytes());
    }
}

fn pad_to_4(buf: &mut Vec<u8>) {
    while buf.len() % 4 != 0 {
        buf.push(0);
    }
}

fn put_name(buf: &mut Vec<u8>, name: &str, format: NcFormat) {
    let bytes = name.as_bytes();
    put_count(buf, bytes.len() as u64, format);
    buf.extend_from_slice(bytes);
    pad_to_4(buf);
}

fn put_absent(buf: &mut Vec<u8>, format: NcFormat) {
    put_u32(buf, ABSENT_TAG);
    put_count(buf, 0, format);
}

fn put_attr(buf: &mut Vec<u8>, attr: &Attr, format: NcFormat) {
    put_name(buf, &attr.name, format);
    put_u32(buf, attr.value.nc_type().code());
    put_count(buf, attr.value.len() as u64, format);
    attr.value.encode_be(buf);
    pad_to_4(buf);
}

fn put_att_list(buf: &mut Vec<u8>, attrs: &[Attr], format: NcFormat) {
    if attrs.is_empty() {
        put_absent(buf, format);
        return;
    }
    put_u32(buf, NC_ATTRIBUTE);
    put_count(buf, attrs.len() as u64, format);
    for attr in attrs {
        put_attr(buf, attr, format);
    }
}

/// Serialize the whole header. Called twice: once with placeholder
/// `begin`s to measure the header's length, once with the real ones.
/// Every field between the two passes is fixed width, so the two lengths
/// are identical by construction -- and the writer asserts it anyway.
pub(crate) fn serialize_header(schema: &Schema, vars: &[VarLayout], numrecs: u64) -> Vec<u8> {
    let format = schema.format;
    let mut buf = Vec::with_capacity(1024);

    buf.extend_from_slice(b"CDF");
    buf.push(format.magic_byte());
    put_count(&mut buf, numrecs, format);

    if schema.dims.is_empty() {
        put_absent(&mut buf, format);
    } else {
        put_u32(&mut buf, NC_DIMENSION);
        put_count(&mut buf, schema.dims.len() as u64, format);
        for dim in &schema.dims {
            put_name(&mut buf, &dim.name, format);
            // The record dimension is written as length 0; its true
            // extent is `numrecs` at the top of the file.
            put_count(&mut buf, if dim.unlimited { 0 } else { dim.len as u64 }, format);
        }
    }

    put_att_list(&mut buf, &schema.gattrs, format);

    if schema.vars.is_empty() {
        put_absent(&mut buf, format);
    } else {
        put_u32(&mut buf, NC_VARIABLE);
        put_count(&mut buf, schema.vars.len() as u64, format);
        for (varid, var) in schema.vars.iter().enumerate() {
            put_name(&mut buf, &var.name, format);
            put_count(&mut buf, var.dimids.len() as u64, format);
            for &dimid in &var.dimids {
                put_count(&mut buf, dimid as u64, format);
            }
            put_att_list(&mut buf, &var.attrs, format);
            put_u32(&mut buf, var.ty.code());
            put_count(&mut buf, vsize_field(vars[varid].vsize_true, format), format);
            put_offset(&mut buf, vars[varid].begin, format);
        }
    }

    buf
}

/// The `vsize` field as netCDF-C writes it.
///
/// The spec documents `vsize` as redundant -- readers recompute the true
/// size from the dimensions -- and as too narrow in CDF-1/CDF-2 to hold a
/// variable larger than `2^32 - 4` bytes, in which case `2^32 - 1` is
/// written as the sentinel. CDF-5 widened the field, so no sentinel.
pub(crate) fn vsize_field(vsize_true: u64, format: NcFormat) -> u64 {
    if format.wide_counts() {
        return vsize_true;
    }
    if vsize_true > (u32::MAX as u64) - 3 {
        u32::MAX as u64
    } else {
        vsize_true
    }
}

/// Where `numrecs` sits, so `finish` can patch it: straight after the
/// 4-byte magic.
pub(crate) const NUMRECS_OFFSET: u64 = 4;

/// The width of the `numrecs` field for this container version.
pub(crate) fn numrecs_bytes(format: NcFormat, numrecs: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8);
    put_count(&mut buf, numrecs, format);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AttrValue, NcType};

    #[test]
    fn pad4_rounds_up() {
        assert_eq!(pad4(0), 0);
        assert_eq!(pad4(1), 4);
        assert_eq!(pad4(3), 4);
        assert_eq!(pad4(4), 4);
        assert_eq!(pad4(5), 8);
    }

    #[test]
    fn vsize_sentinel_only_bites_the_narrow_containers() {
        assert_eq!(vsize_field(24, NcFormat::Offset64), 24);
        assert_eq!(
            vsize_field(u32::MAX as u64 - 3, NcFormat::Offset64),
            u32::MAX as u64 - 3
        );
        assert_eq!(
            vsize_field(u32::MAX as u64 - 2, NcFormat::Offset64),
            u32::MAX as u64
        );
        // CDF-5's field is 64 bits wide, so the true size always fits.
        assert_eq!(
            vsize_field(u32::MAX as u64 + 1000, NcFormat::Cdf5),
            u32::MAX as u64 + 1000
        );
    }

    #[test]
    fn absent_is_eight_bytes_in_cdf2_and_twelve_in_cdf5() {
        let mut buf = Vec::new();
        put_absent(&mut buf, NcFormat::Offset64);
        assert_eq!(buf, vec![0u8; 8]);
        let mut buf = Vec::new();
        put_absent(&mut buf, NcFormat::Cdf5);
        assert_eq!(buf, vec![0u8; 12]);
    }

    #[test]
    fn a_name_is_length_prefixed_and_zero_padded_to_four() {
        let mut buf = Vec::new();
        put_name(&mut buf, "lat_q", NcFormat::Offset64);
        assert_eq!(&buf[..4], &5u32.to_be_bytes());
        assert_eq!(&buf[4..9], b"lat_q");
        assert_eq!(&buf[9..12], &[0, 0, 0], "pad bytes must be zero");
        assert_eq!(buf.len(), 12);
    }

    #[test]
    fn a_text_attribute_counts_bytes_not_elements() {
        let mut buf = Vec::new();
        put_attr(
            &mut buf,
            &Attr {
                name: "units".to_string(),
                value: AttrValue::Text("K".to_string()),
            },
            NcFormat::Offset64,
        );
        // name(4+5+3=12) type(4) nelems(4) "K"+pad(4)
        assert_eq!(buf.len(), 24);
        assert_eq!(&buf[12..16], &NcType::Char.code().to_be_bytes());
        assert_eq!(&buf[16..20], &1u32.to_be_bytes());
        assert_eq!(buf[20], b'K');
        assert_eq!(&buf[21..24], &[0, 0, 0]);
    }
}
