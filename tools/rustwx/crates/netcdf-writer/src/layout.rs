//! Where every byte goes.
//!
//! A classic file's data section is two regions in this order:
//!
//! 1. the **fixed** variables, one contiguous padded block each, in
//!    definition order;
//! 2. the **record** section, `numrecs` records back to back, where one
//!    record holds one slab from every record variable, again in
//!    definition order.
//!
//! A record variable's `begin` is the offset of its slab inside record 0,
//! so record `r` of variable `v` lives at `begin_v + r * recsize`.
//!
//! **The special case, MEASURED against netCDF-C** (netCDF4-python 1.7.4,
//! 2026-08-16): when exactly ONE variable uses the record dimension, the
//! record stride is the RAW slab size, not the 4-byte-padded one, while
//! the header's `vsize` field still carries the padded value. A 3-byte
//! `NC_CHAR` slab wrote `vsize = 4` and a stride of 3. netCDF-C spells it
//! in `nc3internal.c`'s `NC_computeshapes`: *"for special case of exactly
//! one record variable, pack value"*. Getting it wrong is not a crash --
//! the reader walks the record section with the wrong stride and hands
//! back shifted bytes.

use crate::error::{NcWriteError, Result};
use crate::header::{pad4, serialize_header};
use crate::schema::Schema;
use crate::types::NcFormat;

/// Per-variable byte geometry.
#[derive(Debug, Clone)]
pub(crate) struct VarLayout {
    /// Elements in the whole variable (fixed) or in one record slab
    /// (record variable).
    pub elems: u64,
    /// `elems * type size`, unpadded.
    pub slab_bytes: u64,
    /// The padded size the header's `vsize` field is derived from.
    pub vsize_true: u64,
    /// File offset: absolute for a fixed variable, offset of the slab
    /// inside record 0 for a record variable.
    pub begin: u64,
    pub is_record: bool,
    /// Index among the record variables only (used for bookkeeping).
    pub record_index: Option<usize>,
}

/// The whole file's geometry.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    pub vars: Vec<VarLayout>,
    pub header_len: u64,
    /// Where record 0 starts.
    pub rec_base: u64,
    /// Bytes from the start of one record to the start of the next.
    pub recsize: u64,
    pub num_record_vars: usize,
    /// Are individual slabs padded to 4 bytes? False in the
    /// one-record-variable special case.
    pub pad_record_slabs: bool,
}

impl Layout {
    pub(crate) fn compute(schema: &Schema) -> Result<Layout> {
        let format = schema.format;
        let mut vars: Vec<VarLayout> = Vec::with_capacity(schema.vars.len());
        let mut num_record_vars = 0usize;

        for (varid, var) in schema.vars.iter().enumerate() {
            let is_record = schema.is_record_var(varid);
            // A record variable's geometry is one record's worth: skip
            // the leading record dimension.
            let counted = if is_record { &var.dimids[1..] } else { &var.dimids[..] };
            let mut elems: u64 = 1;
            for &dimid in counted {
                let len = schema.dims[dimid].len as u64;
                elems = elems.checked_mul(len).ok_or_else(|| {
                    NcWriteError::Capacity(format!(
                        "variable '{}' element count overflows 64 bits",
                        var.name
                    ))
                })?;
            }
            let slab_bytes = elems.checked_mul(var.ty.size() as u64).ok_or_else(|| {
                NcWriteError::Capacity(format!(
                    "variable '{}' byte size overflows 64 bits",
                    var.name
                ))
            })?;
            let record_index = if is_record {
                let index = num_record_vars;
                num_record_vars += 1;
                Some(index)
            } else {
                None
            };
            vars.push(VarLayout {
                elems,
                slab_bytes,
                vsize_true: pad4(slab_bytes),
                begin: 0,
                is_record,
                record_index,
            });
        }

        // The one-record-variable special case.
        let pad_record_slabs = num_record_vars != 1;

        // Pass 1: measure the header with placeholder begins. Every field
        // is fixed width, so the length does not move in pass 2.
        let header_len = serialize_header(schema, &vars, 0).len() as u64;

        // Fixed variables first, each padded.
        let mut cursor = header_len;
        for layout in vars.iter_mut().filter(|v| !v.is_record) {
            layout.begin = cursor;
            cursor = cursor.checked_add(pad4(layout.slab_bytes)).ok_or_else(|| {
                NcWriteError::Capacity("fixed data section size overflows 64 bits".to_string())
            })?;
        }
        let rec_base = cursor;

        // Then the record section: slab offsets inside record 0.
        let mut offset_in_record = 0u64;
        for layout in vars.iter_mut().filter(|v| v.is_record) {
            layout.begin = rec_base + offset_in_record;
            let step = if pad_record_slabs {
                pad4(layout.slab_bytes)
            } else {
                layout.slab_bytes
            };
            offset_in_record = offset_in_record.checked_add(step).ok_or_else(|| {
                NcWriteError::Capacity("record size overflows 64 bits".to_string())
            })?;
        }
        let recsize = offset_in_record;

        let layout = Layout {
            vars,
            header_len,
            rec_base,
            recsize,
            num_record_vars,
            pad_record_slabs,
        };
        layout.check_offset_field_width(schema, format)?;
        layout.check_variable_size_limits(schema, format)?;
        Ok(layout)
    }

    /// CDF-1 and CDF-2 keep `vsize` in 32 bits, and the format documents
    /// the `2^32 - 1` sentinel as usable for at most ONE variable: the
    /// last fixed-size variable when there are no record variables, or
    /// the last record variable. A second sentinel makes two variables
    /// indistinguishable in the header, and netCDF-C refuses to open the
    /// file ("NetCDF: One or more variable sizes violate format
    /// constraints"). Refuse here, and name CDF-5 as the container that
    /// has the room.
    fn check_variable_size_limits(&self, schema: &Schema, format: NcFormat) -> Result<()> {
        if format.wide_counts() {
            return Ok(());
        }
        const LIMIT: u64 = u32::MAX as u64 - 3;

        let last_fixed = self.vars.iter().rposition(|v| !v.is_record);
        let last_record = self.vars.iter().rposition(|v| v.is_record);
        for (varid, layout) in self.vars.iter().enumerate() {
            if layout.slab_bytes <= LIMIT {
                continue;
            }
            let exempt = if layout.is_record {
                last_record == Some(varid)
            } else {
                last_fixed == Some(varid) && self.num_record_vars == 0
            };
            if exempt {
                continue;
            }
            let what = if layout.is_record {
                "one record of variable"
            } else {
                "variable"
            };
            return Err(NcWriteError::Capacity(format!(
                "{what} '{}' needs {} bytes, past the {} limit of {LIMIT}; only the \
                 last such variable may exceed it. Write this file as CDF-5 \
                 (NcFormat::Cdf5), or move the variable last",
                schema.vars[varid].name,
                layout.slab_bytes,
                format.label(),
            )));
        }
        Ok(())
    }

    /// CDF-1 keeps `begin` in 32 bits. A data section past that wraps the
    /// offset and every variable after the wrap reads from the wrong
    /// place -- silently, because a wrapped offset is still a valid
    /// number. Refuse, and name the container that has the room.
    fn check_offset_field_width(&self, schema: &Schema, format: NcFormat) -> Result<()> {
        if format.wide_offsets() {
            return Ok(());
        }
        let limit = u32::MAX as u64;
        for (varid, layout) in self.vars.iter().enumerate() {
            if layout.begin > limit {
                return Err(NcWriteError::Capacity(format!(
                    "variable '{}' starts at byte {} which does not fit CDF-1's \
                     32-bit begin field; write this file as CDF-2 \
                     (NcFormat::Offset64) or CDF-5",
                    schema.vars[varid].name, layout.begin
                )));
            }
        }
        // The end of the fixed section, and of record 0, must also land
        // inside the addressable range for the file to be readable.
        let end = self.rec_base.saturating_add(self.recsize);
        if end > limit {
            return Err(NcWriteError::Capacity(format!(
                "the data section ends at byte {end}, past CDF-1's 32-bit offset \
                 limit of {limit}; write this file as CDF-2 (NcFormat::Offset64) \
                 or CDF-5"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NcType;

    #[test]
    fn fixed_variables_are_laid_out_in_definition_order_after_the_header() {
        let mut schema = Schema::new(NcFormat::Offset64);
        let n = schema.def_dim("n", 3, false).unwrap();
        let a = schema.def_var("a", NcType::Float, &[n]).unwrap();
        let b = schema.def_var("b", NcType::Double, &[n]).unwrap();
        let layout = Layout::compute(&schema).unwrap();
        assert_eq!(layout.vars[a].begin, layout.header_len);
        assert_eq!(layout.vars[b].begin, layout.header_len + 12);
        assert_eq!(layout.rec_base, layout.header_len + 12 + 24);
        assert_eq!(layout.recsize, 0);
    }

    #[test]
    fn one_record_variable_gives_an_unpadded_stride() {
        let mut schema = Schema::new(NcFormat::Offset64);
        let t = schema.def_dim("t", 0, true).unwrap();
        let s = schema.def_dim("s", 3, false).unwrap();
        let c = schema.def_var("c", NcType::Char, &[t, s]).unwrap();
        let layout = Layout::compute(&schema).unwrap();
        assert_eq!(layout.num_record_vars, 1);
        assert!(!layout.pad_record_slabs);
        assert_eq!(layout.recsize, 3, "stride is the RAW slab size");
        assert_eq!(
            layout.vars[c].vsize_true, 4,
            "vsize is still the padded size"
        );
    }

    #[test]
    fn two_record_variables_give_a_padded_stride() {
        let mut schema = Schema::new(NcFormat::Offset64);
        let t = schema.def_dim("t", 0, true).unwrap();
        let s = schema.def_dim("s", 3, false).unwrap();
        let c = schema.def_var("c", NcType::Char, &[t, s]).unwrap();
        let f = schema.def_var("f", NcType::Float, &[t]).unwrap();
        let layout = Layout::compute(&schema).unwrap();
        assert!(layout.pad_record_slabs);
        assert_eq!(layout.recsize, 8);
        assert_eq!(layout.vars[c].begin, layout.rec_base);
        assert_eq!(layout.vars[f].begin, layout.rec_base + 4);
    }

    #[test]
    fn record_slabs_skip_the_leading_record_dimension() {
        let mut schema = Schema::new(NcFormat::Offset64);
        let t = schema.def_dim("Time", 0, true).unwrap();
        let z = schema.def_dim("bottom_top", 50, false).unwrap();
        let y = schema.def_dim("south_north", 3, false).unwrap();
        let v = schema.def_var("T", NcType::Float, &[t, z, y]).unwrap();
        let layout = Layout::compute(&schema).unwrap();
        assert_eq!(layout.vars[v].elems, 150);
        assert_eq!(layout.vars[v].slab_bytes, 600);
    }

    #[test]
    fn cdf2_allows_one_oversized_record_variable_but_not_two() {
        // 2^30 doubles = 8 GiB per record slab, past the 32-bit vsize
        // field.  One is legal (it is last and takes the sentinel); two
        // is a file netCDF-C refuses to open.
        let build = |count: usize| {
            let mut schema = Schema::new(NcFormat::Offset64);
            let t = schema.def_dim("t", 0, true).unwrap();
            let big = schema.def_dim("big", 1 << 30, false).unwrap();
            for index in 0..count {
                schema
                    .def_var(&format!("v{index}"), NcType::Double, &[t, big])
                    .unwrap();
            }
            Layout::compute(&schema)
        };
        assert!(build(1).is_ok(), "one oversized record variable is legal");
        let err = build(2).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("v0"), "the offending variable is named: {msg}");
        assert!(msg.contains("CDF-5"), "got: {msg}");
    }

    #[test]
    fn cdf5_has_room_for_every_oversized_variable() {
        let mut schema = Schema::new(NcFormat::Cdf5);
        let t = schema.def_dim("t", 0, true).unwrap();
        let big = schema.def_dim("big", 1 << 30, false).unwrap();
        schema.def_var("a", NcType::Double, &[t, big]).unwrap();
        schema.def_var("b", NcType::Double, &[t, big]).unwrap();
        assert!(Layout::compute(&schema).is_ok());
    }

    #[test]
    fn a_scalar_variable_is_one_element() {
        let mut schema = Schema::new(NcFormat::Offset64);
        let v = schema.def_var("gain", NcType::Double, &[]).unwrap();
        let layout = Layout::compute(&schema).unwrap();
        assert_eq!(layout.vars[v].elems, 1);
        assert_eq!(layout.vars[v].slab_bytes, 8);
    }
}
