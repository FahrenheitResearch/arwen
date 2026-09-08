//! Map HDF5 datatypes to NetCDF-4 types.
//!
//! HDF5 has a richer type system than NetCDF-4. This module maps the subset
//! of HDF5 types that are valid in NetCDF-4 to `NcType`:
//!
//! | HDF5 Datatype                 | NcType   |
//! |-------------------------------|----------|
//! | FixedPoint { size=1, signed } | Byte     |
//! | FixedPoint { size=1, !signed} | UByte    |
//! | FixedPoint { size=2, signed } | Short    |
//! | FixedPoint { size=2, !signed} | UShort   |
//! | FixedPoint { size=4, signed } | Int      |
//! | FixedPoint { size=4, !signed} | UInt     |
//! | FixedPoint { size=8, signed } | Int64    |
//! | FixedPoint { size=8, !signed} | UInt64   |
//! | FloatingPoint { size=4 }      | Float    |
//! | FloatingPoint { size=8 }      | Double   |
//! | String (any)                  | String   |
//! | Enum { base=byte }            | Byte*    |
//! | Compound { .. }               | Compound |
//! | Opaque { .. }                 | Opaque   |
//! | Array { base, dims }          | Array    |
//! | VarLen { base=u8 }            | String*  |
//! | VarLen { base!=u8 }           | VLen     |
//!
//! * Enums are mapped to their base integer type.
//! * NetCDF-4 `NC_STRING` is stored as HDF5 variable-length bytes.

use hdf5_reader::messages::datatype::Datatype;

use crate::error::{Error, Result};
use crate::types::{NcCompoundField, NcType};

/// Map an HDF5 datatype to a NetCDF type.
pub fn hdf5_to_nc_type(dtype: &Datatype) -> Result<NcType> {
    match dtype {
        Datatype::FixedPoint { size, signed, .. } => match (size, signed) {
            (1, true) => Ok(NcType::Byte),
            (1, false) => Ok(NcType::UByte),
            (2, true) => Ok(NcType::Short),
            (2, false) => Ok(NcType::UShort),
            (4, true) => Ok(NcType::Int),
            (4, false) => Ok(NcType::UInt),
            (8, true) => Ok(NcType::Int64),
            (8, false) => Ok(NcType::UInt64),
            _ => Err(Error::InvalidData(format!(
                "unsupported HDF5 integer size {} for NetCDF-4",
                size
            ))),
        },
        Datatype::FloatingPoint { size, .. } => match size {
            4 => Ok(NcType::Float),
            8 => Ok(NcType::Double),
            _ => Err(Error::InvalidData(format!(
                "unsupported HDF5 float size {} for NetCDF-4",
                size
            ))),
        },
        Datatype::String { .. } => Ok(NcType::String),
        Datatype::Enum { base, .. } => {
            // Enums map to their base integer type.
            hdf5_to_nc_type(base)
        }
        Datatype::Compound { size, fields } => {
            let mut nc_fields = Vec::with_capacity(fields.len());
            for f in fields {
                nc_fields.push(NcCompoundField {
                    name: f.name.clone(),
                    offset: f.byte_offset as u64,
                    dtype: hdf5_to_nc_type(&f.datatype)?,
                });
            }
            Ok(NcType::Compound {
                size: *size,
                fields: nc_fields,
            })
        }
        Datatype::Opaque { size, tag } => Ok(NcType::Opaque {
            size: *size,
            tag: tag.clone(),
        }),
        Datatype::Array { base, dims } => {
            let base_nc = hdf5_to_nc_type(base)?;
            Ok(NcType::Array {
                base: Box::new(base_nc),
                dims: dims.clone(),
            })
        }
        Datatype::VarLen { base }
            if matches!(
                base.as_ref(),
                Datatype::FixedPoint {
                    size: 1,
                    signed: false,
                    ..
                }
            ) =>
        {
            Ok(NcType::String)
        }
        Datatype::VarLen { base } => {
            let base_nc = hdf5_to_nc_type(base)?;
            Ok(NcType::VLen {
                base: Box::new(base_nc),
            })
        }
        _ => Err(Error::InvalidData(format!(
            "HDF5 datatype {:?} has no NetCDF-4 equivalent",
            dtype
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5_reader::error::ByteOrder;

    #[test]
    fn test_integer_types() {
        let bo = ByteOrder::LittleEndian;
        assert_eq!(
            hdf5_to_nc_type(&Datatype::FixedPoint {
                size: 1,
                signed: true,
                byte_order: bo
            })
            .unwrap(),
            NcType::Byte
        );
        assert_eq!(
            hdf5_to_nc_type(&Datatype::FixedPoint {
                size: 1,
                signed: false,
                byte_order: bo
            })
            .unwrap(),
            NcType::UByte
        );
        assert_eq!(
            hdf5_to_nc_type(&Datatype::FixedPoint {
                size: 4,
                signed: true,
                byte_order: bo
            })
            .unwrap(),
            NcType::Int
        );
        assert_eq!(
            hdf5_to_nc_type(&Datatype::FixedPoint {
                size: 8,
                signed: false,
                byte_order: bo
            })
            .unwrap(),
            NcType::UInt64
        );
    }

    #[test]
    fn test_float_types() {
        let bo = ByteOrder::LittleEndian;
        assert_eq!(
            hdf5_to_nc_type(&Datatype::FloatingPoint {
                size: 4,
                byte_order: bo
            })
            .unwrap(),
            NcType::Float
        );
        assert_eq!(
            hdf5_to_nc_type(&Datatype::FloatingPoint {
                size: 8,
                byte_order: bo
            })
            .unwrap(),
            NcType::Double
        );
    }

    #[test]
    fn test_varlen_u8_maps_to_string() {
        let bo = ByteOrder::LittleEndian;
        assert_eq!(
            hdf5_to_nc_type(&Datatype::VarLen {
                base: Box::new(Datatype::FixedPoint {
                    size: 1,
                    signed: false,
                    byte_order: bo,
                }),
            })
            .unwrap(),
            NcType::String
        );
    }
}
