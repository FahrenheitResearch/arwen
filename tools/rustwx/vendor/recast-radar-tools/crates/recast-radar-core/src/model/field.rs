//! Dataset variables: compact row-major storage with CF packing
//! (`docs/design/fm301-model.md` sections 4 and 7).

use std::borrow::Cow;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::names::{FieldName, Polarization, Quantity};
use super::sweep::RangeCoord;
use super::values::{ArrayBuf, AttrValue, Scalar};

/// Smallest 16-bit field [`Field::to_physical`] decodes through a 65,536-entry
/// table: below it, resolving each value costs less than building the table.
const LUT16_MIN_VALUES: usize = 1 << 17;

/// One dataset variable of a sweep: `<name>(time, range)` in FM301.
///
/// Values are stored row-major `[nrays × ngates]` in the source's encoding and
/// in the field's native gate geometry. Row `r` belongs to ray `r` of the sweep.
///
/// With the `serde` feature, deserializing a field checks that `data` holds
/// `nrays × ngates` values and that `absent_rows` ascend below `nrays`, and
/// fails when a check does.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "super::serde_checked::FieldRepr")
)]
pub struct Field {
    /// Variable name in the sweep group (section 8).
    pub name: FieldName,
    /// Semantic class regardless of spelling (DBZH, DBZ, DBZHC_F all ->
    /// Reflectivity).
    pub quantity: Quantity,
    /// Polarization channel of the field (from its name and standard name).
    pub polarization: Polarization,
    /// CF / FM301 attributes: `standard_name`, `long_name`, `units`, Table
    /// 301-10.
    pub attrs: FieldAttrs,
    /// Rows. Equal to the sweep's ray count once the sweep is sealed.
    pub nrays: u32,
    /// Native gates per row.
    pub ngates: u32,
    /// Where the native gates sit on the sweep's `range` coordinate (section 6).
    pub gates: GateMapping,
    /// Row-major `[nrays × ngates]` values in the source encoding (section 7).
    pub data: FieldData,
    /// Rows the source did not provide for this field, ascending. Each is filled
    /// with the coding's fill code (NaN for floats without one). Empty in the
    /// common case, which costs no allocation.
    pub absent_rows: Vec<u32>,
}

/// Native gate `i` covers sweep-range gates
/// `start + i*stride ..= start + i*stride + stride - 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GateMapping {
    /// First sweep-range gate that native gate 0 covers.
    pub start: u32,
    /// 1: native spacing equals the sweep range spacing. 4: 1 km gates over a
    /// 250 m range.
    pub stride: u32,
}

impl GateMapping {
    /// `start = 0`, `stride = 1`.
    pub const IDENTITY: GateMapping = GateMapping {
        start: 0,
        stride: 1,
    };

    /// One past the last sweep-range gate covered by `ngates` native gates.
    pub fn end(self, ngates: u32) -> Option<u64> {
        u64::from(self.start).checked_add(u64::from(ngates) * u64::from(self.stride.max(1)))
    }
}

impl Default for GateMapping {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Field values in their source encoding with their coding.
///
/// Not `#[non_exhaustive]`, on purpose: the variants are the storage types of
/// the data model, and code that reads raw storage (decoders, renderers,
/// writers, bindings) must handle every one. A new storage type is a
/// breaking change that every such `match` has to see. [`RowRef`] and
/// [`Coding`] mirror these variants and are exhaustive for the same reason,
/// as are [`LinearTransform`] and [`FloatWidth`], which the same code needs
/// to turn packed values into physical ones or to write the packing back.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FieldData {
    /// Unsigned 8-bit codes (NEXRAD moments, most ODIM planes).
    U8 {
        /// Row-major `[nrays × ngates]` codes.
        values: Vec<u8>,
        /// How codes map to physical values and sentinels.
        coding: IntCoding<u8>,
    },
    /// Unsigned 16-bit codes (NEXRAD ZDR and PHIDP, 16-bit ODIM planes).
    U16 {
        /// Row-major `[nrays × ngates]` codes.
        values: Vec<u16>,
        /// How codes map to physical values and sentinels.
        coding: IntCoding<u16>,
    },
    /// Signed 8-bit codes (CfRadial `byte` fields).
    I8 {
        /// Row-major `[nrays × ngates]` codes.
        values: Vec<i8>,
        /// How codes map to physical values and sentinels.
        coding: IntCoding<i8>,
    },
    /// Signed 16-bit codes (CfRadial `short` fields, DORADE).
    I16 {
        /// Row-major `[nrays × ngates]` codes.
        values: Vec<i16>,
        /// How codes map to physical values and sentinels.
        coding: IntCoding<i16>,
    },
    /// 32-bit integer sources (CfRadial `int` fields), stored verbatim.
    I32 {
        /// Row-major `[nrays × ngates]` codes.
        values: Vec<i32>,
        /// How codes map to physical values and sentinels.
        coding: IntCoding<i32>,
    },
    /// Physical values (derived products, float32 sources), stored verbatim.
    F32 {
        /// Row-major `[nrays × ngates]` values.
        values: Vec<f32>,
        /// Fill and undetect values, and an optional gain and offset.
        coding: FloatCoding<f32>,
    },
    /// float64 sources (ODIM float64 planes, CfRadial `double` fields), stored
    /// verbatim.
    F64 {
        /// Row-major `[nrays × ngates]` values.
        values: Vec<f64>,
        /// Fill and undetect values, and an optional gain and offset.
        coding: FloatCoding<f64>,
    },
}

/// Coding of an integer field.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IntCoding<T> {
    /// How a code maps to a physical value.
    pub transform: LinearTransform,
    /// CF `_FillValue`: the code for "no data" and the code the view pads with.
    /// NEXRAD 0, ODIM `nodata`, CfRadial `_FillValue`.
    pub fill_value: Option<T>,
    /// FM301 Table 301-10 `_Undetect`: radiated, but no valid echo. NEXRAD 0
    /// (below threshold, equal to `fill_value`), ODIM `undetect`.
    pub undetect: Option<T>,
    /// NEXRAD 1. Exported as `flag_values = [1]`, `flag_meanings =
    /// "range_folded"`.
    pub range_folded: Option<T>,
    /// CF `valid_range` in packed units (WMO-CF.5.2.15).
    pub valid_range: Option<[T; 2]>,
}

impl<T: PackedInt> IntCoding<T> {
    /// A coding with `transform` and no sentinels.
    pub fn new(transform: LinearTransform) -> Self {
        Self {
            transform,
            fill_value: None,
            undetect: None,
            range_folded: None,
            valid_range: None,
        }
    }

    /// NEXRAD Level II moment coding (section 7.1): ICD `value = (raw -
    /// offset) / scale`, raw 0 below threshold (`_FillValue` and `_Undetect`),
    /// raw 1 range folded, `valid_range = [2, MAX]`.
    pub fn nexrad(scale: f32, offset: f32) -> Self {
        Self {
            transform: LinearTransform::IcdScaleOffset { scale, offset },
            fill_value: Some(T::from_u8(0)),
            undetect: Some(T::from_u8(0)),
            range_folded: Some(T::from_u8(1)),
            valid_range: Some([T::from_u8(2), T::MAX]),
        }
    }

    /// The code used to fill padding and absent rows: `fill_value`, else 0.
    pub fn fill_code(&self) -> T {
        self.fill_value.unwrap_or(T::from_u8(0))
    }

    /// Resolve one raw code (section 4, `Field::gate` rules 2 to 6). A code
    /// the transform gives no value for (a [`LevelTable`] level without a
    /// value) is `Missing`.
    pub fn resolve(&self, raw: T) -> Gate {
        if self.undetect == Some(raw) {
            Gate::Undetect
        } else if self.fill_value == Some(raw) {
            Gate::Missing
        } else if self.range_folded == Some(raw) {
            Gate::RangeFolded
        } else if matches!(self.valid_range, Some([lo, hi]) if raw < lo || raw > hi) {
            Gate::Missing
        } else {
            let value = self.transform.apply(raw.as_f64());
            if value.is_nan() {
                Gate::Missing
            } else {
                Gate::Value(value)
            }
        }
    }
}

/// The transform from packed codes to physical values.
///
/// Every variant but [`LinearTransform::Levels`] is linear and has CF
/// `scale_factor` / `add_offset` equivalents ([`LinearTransform::is_linear`]).
/// `Levels` covers the NEXRAD Level III data level encodings that are not a
/// linear function of the code (16-level threshold tables, high resolution
/// VIL, enhanced echo tops); CF has no attribute for them, so the FM301 view
/// writes such fields decoded, with the codes beside them (`crate::fm301`).
///
/// Exhaustive, like [`FieldData`]: a writer or binding that converts or
/// re-encodes packed values must handle every transform, so a new one is a
/// breaking change instead of a case a wildcard arm would silently mishandle.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LinearTransform {
    /// `physical = (raw - offset) / scale`, evaluated in f32. NEXRAD ICD form;
    /// Py-ART evaluates exactly this expression. The
    /// view writes `scale_factor = 1/scale` and `add_offset = -offset/scale` as
    /// float64, as xradar does.
    IcdScaleOffset {
        /// The ICD scale (codes per physical unit).
        scale: f32,
        /// The ICD offset, in codes.
        offset: f32,
    },
    /// `physical = raw * scale_factor + add_offset`, evaluated in f64. CF and
    /// ODIM gain-offset form. `attr_width` is the type the source wrote the two
    /// attributes in; xarray derives the decoded dtype from it.
    CfScaleOffset {
        /// CF `scale_factor` (ODIM `gain`).
        scale_factor: f64,
        /// CF `add_offset` (ODIM `offset`).
        add_offset: f64,
        /// The type the source wrote the two attributes in.
        attr_width: FloatWidth,
    },
    /// `physical = table.value(raw)`: a data level encoding that is not
    /// linear (NEXRAD Level III). NaN where a level has no value, which
    /// [`IntCoding::resolve`] reports as `Missing`.
    Levels(LevelTable),
}

impl LinearTransform {
    /// Apply to a raw value. Integer codes and f32 values are exact in f64.
    #[inline]
    pub fn apply(self, raw: f64) -> f32 {
        match self {
            Self::IcdScaleOffset { scale, offset } => (raw as f32 - offset) / scale,
            Self::CfScaleOffset {
                scale_factor,
                add_offset,
                ..
            } => (raw * scale_factor + add_offset) as f32,
            Self::Levels(table) => table.value(raw),
        }
    }

    /// `true` for the transforms CF `scale_factor` / `add_offset` express
    /// exactly; `false` for [`LinearTransform::Levels`].
    pub fn is_linear(self) -> bool {
        !matches!(self, Self::Levels(_))
    }

    /// CF `scale_factor`, or `None` for [`LinearTransform::Levels`], which
    /// has none: a writer must store such a field decoded (or its codes with
    /// the level table), never packed with a scale factor.
    pub fn scale_factor(self) -> Option<f64> {
        match self {
            Self::IcdScaleOffset { scale, .. } => Some(1.0 / f64::from(scale)),
            Self::CfScaleOffset { scale_factor, .. } => Some(scale_factor),
            Self::Levels(_) => None,
        }
    }

    /// CF `add_offset`, or `None` for [`LinearTransform::Levels`], which has
    /// none (see [`scale_factor`](Self::scale_factor)).
    pub fn add_offset(self) -> Option<f64> {
        match self {
            Self::IcdScaleOffset { scale, offset } => Some(-f64::from(offset) / f64::from(scale)),
            Self::CfScaleOffset { add_offset, .. } => Some(add_offset),
            Self::Levels(_) => None,
        }
    }

    /// The type CF `scale_factor` / `add_offset` attributes are written in
    /// (F32 for [`LinearTransform::Levels`], which writes none).
    pub fn attr_width(self) -> FloatWidth {
        match self {
            Self::IcdScaleOffset { .. } => FloatWidth::F64,
            Self::CfScaleOffset { attr_width, .. } => attr_width,
            Self::Levels(_) => FloatWidth::F32,
        }
    }
}

/// A NEXRAD Level III data level encoding that is not a linear function of
/// the level (ICD 2620001 Figure 3-6 sheet 6 Note 1). Every variant is
/// evaluated in f64 and rounded to f32 once. Code that only decodes needs
/// [`LevelTable::value`], not a `match`; later encodings may add variants.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum LevelTable {
    /// A 16-level product: level `n` (0-15) is `values[n]` (the threshold
    /// value its halfword gives), NaN for a level without a value. Levels
    /// from 16 have no value.
    Sixteen([f32; 16]),
    /// `(raw & mask) / scale - offset`: the bits outside `mask` are flags
    /// (enhanced echo tops, product 135: mask 0x7F, bit 0x80 "topped",
    /// described by the field's `flag_masks`).
    Masked {
        /// Data bits.
        mask: u16,
        /// Divisor.
        scale: f32,
        /// Offset subtracted after dividing.
        offset: f32,
    },
    /// High resolution VIL (product 134): `(raw - linear_offset) /
    /// linear_scale` below `log_start`, `exp((raw - log_offset) /
    /// log_scale)` from it.
    LinearLog {
        /// Linear scale.
        linear_scale: f32,
        /// Linear offset.
        linear_offset: f32,
        /// First level of the logarithmic part.
        log_start: u16,
        /// Log scale.
        log_scale: f32,
        /// Log offset.
        log_offset: f32,
    },
}

impl LevelTable {
    /// The value of level `raw`; NaN when the level has none or the result
    /// is not finite.
    pub fn value(self, raw: f64) -> f32 {
        let value = match self {
            Self::Sixteen(values) => {
                return if (0.0..16.0).contains(&raw) {
                    values[raw as usize]
                } else {
                    f32::NAN
                };
            }
            Self::Masked {
                mask,
                scale,
                offset,
            } => {
                let data = (raw as u32) & u32::from(mask);
                f64::from(data) / f64::from(scale) - f64::from(offset)
            }
            Self::LinearLog {
                linear_scale,
                linear_offset,
                log_start,
                log_scale,
                log_offset,
            } => {
                if raw < f64::from(log_start) {
                    (raw - f64::from(linear_offset)) / f64::from(linear_scale)
                } else {
                    ((raw - f64::from(log_offset)) / f64::from(log_scale)).exp()
                }
            }
        };
        if value.is_finite() {
            value as f32
        } else {
            f32::NAN
        }
    }
}

/// The float type a source wrote `scale_factor` and `add_offset` in.
///
/// Exhaustive, like [`LinearTransform`]: a writer chooses the attribute type
/// from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FloatWidth {
    /// `float` (32-bit).
    F32,
    /// `double` (64-bit).
    F64,
}

/// Coding of a float field. Values are stored as the source wrote them. NaN is
/// always missing. A non-NaN source fill (for example -9999) stays in the data
/// and resolves to `Missing`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FloatCoding<T> {
    /// `None`: the values are physical. `Some`: a plane whose gain / offset is
    /// not 1 / 0, applied on read.
    pub transform: Option<LinearTransform>,
    /// CF `_FillValue`: the value for "no data". NaN is always missing as well.
    pub fill_value: Option<T>,
    /// FM301 `_Undetect`: radiated, but no valid echo (ODIM `undetect`).
    pub undetect: Option<T>,
}

/// Integer types a field can be packed in.
pub trait PackedInt: Copy + PartialEq + PartialOrd + std::fmt::Debug + private::Sealed {
    /// The largest value of the type.
    const MAX: Self;
    /// NumPy-style dtype name (`uint8`, `int16`, ...).
    const DTYPE: &'static str;
    /// The value of a `u8` code in this type.
    fn from_u8(value: u8) -> Self;
    /// The value as `f64` (exact for every packed type).
    fn as_f64(self) -> f64;
    /// The value as a typed scalar.
    fn to_scalar(self) -> Scalar;
    /// A typed array of this integer type.
    fn array(values: Vec<Self>) -> ArrayBuf;
    /// Narrow an `i64`, `None` when it does not fit.
    fn from_i64(value: i64) -> Option<Self>;
}

mod private {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
}

macro_rules! packed_int {
    ($t:ty, $variant:ident, $dtype:literal) => {
        impl PackedInt for $t {
            const MAX: Self = <$t>::MAX;
            const DTYPE: &'static str = $dtype;
            #[inline]
            fn from_u8(value: u8) -> Self {
                value as $t
            }
            #[inline]
            fn as_f64(self) -> f64 {
                f64::from(self)
            }
            fn to_scalar(self) -> Scalar {
                Scalar::$variant(self)
            }
            fn array(values: Vec<Self>) -> ArrayBuf {
                ArrayBuf::$variant(values)
            }
            fn from_i64(value: i64) -> Option<Self> {
                <$t>::try_from(value).ok()
            }
        }
    };
}

packed_int!(u8, U8, "uint8");
packed_int!(u16, U16, "uint16");
packed_int!(i8, I8, "int8");
packed_int!(i16, I16, "int16");
packed_int!(i32, I32, "int32");

fn resolve_float(
    raw: f64,
    transform: Option<LinearTransform>,
    is_fill: bool,
    is_undetect: bool,
) -> Gate {
    if raw.is_nan() {
        Gate::Missing
    } else if is_undetect {
        Gate::Undetect
    } else if is_fill {
        Gate::Missing
    } else {
        Gate::Value(match transform {
            Some(transform) => transform.apply(raw),
            None => raw as f32,
        })
    }
}

impl FloatCoding<f32> {
    /// Resolve a stored value: `Undetect` when it equals `undetect`, `Missing`
    /// when it is NaN or equals `fill_value`, else its physical value (through
    /// `transform` when there is one).
    pub fn resolve(&self, raw: f32) -> Gate {
        resolve_float(
            f64::from(raw),
            self.transform,
            self.fill_value
                .is_some_and(|fill| fill.to_bits() == raw.to_bits()),
            self.undetect
                .is_some_and(|undetect| undetect.to_bits() == raw.to_bits()),
        )
    }

    /// The code used to fill padding and absent rows: `fill_value`, else NaN.
    pub fn fill_code(&self) -> f32 {
        self.fill_value.unwrap_or(f32::NAN)
    }
}

impl FloatCoding<f64> {
    /// Resolve a stored value: `Undetect` when it equals `undetect`, `Missing`
    /// when it is NaN or equals `fill_value`, else its physical value (through
    /// `transform` when there is one).
    pub fn resolve(&self, raw: f64) -> Gate {
        resolve_float(
            raw,
            self.transform,
            self.fill_value
                .is_some_and(|fill| fill.to_bits() == raw.to_bits()),
            self.undetect
                .is_some_and(|undetect| undetect.to_bits() == raw.to_bits()),
        )
    }

    /// The code used to fill padding and absent rows: `fill_value`, else NaN.
    pub fn fill_code(&self) -> f64 {
        self.fill_value.unwrap_or(f64::NAN)
    }
}

/// CF and FM301 attributes of a field. `None` / empty means "not stated by the
/// source"; the FM301 view fills `standard_name`, `long_name` and `units` of
/// known names from the name table.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FieldAttrs {
    /// CF `standard_name`.
    pub standard_name: Option<Cow<'static, str>>,
    /// CF `long_name`.
    pub long_name: Option<Cow<'static, str>>,
    /// UDUNITS `units` of the physical values.
    pub units: Option<Cow<'static, str>>,
    // Table 301-10, all optional
    /// `sampling_ratio`: samples per gate over nominal (Table 301-10).
    pub sampling_ratio: Option<f32>,
    /// `is_discrete`: the values are categories, not a continuous quantity.
    pub is_discrete: Option<bool>,
    /// `field_folds`: the values fold (velocity, phase) between the fold limits.
    pub field_folds: Option<bool>,
    /// `fold_limit_lower`: lower fold limit, physical units.
    pub fold_limit_lower: Option<f32>,
    /// `fold_limit_upper`: upper fold limit, physical units.
    pub fold_limit_upper: Option<f32>,
    /// `is_quality_field`: the field describes the quality of other fields.
    pub is_quality_field: Option<bool>,
    /// `qualified_variables`: the fields a quality field describes.
    pub qualified_variables: Vec<FieldName>,
    /// `ancillary_variables`: the quality fields that describe this field.
    pub ancillary_variables: Vec<FieldName>,
    /// `thresholding_xml`: the thresholding applied to the field, as XML.
    pub thresholding_xml: Option<String>,
    /// `flag_values`, `flag_masks` and `flag_meanings` of discrete fields,
    /// besides the range-folded flag, which comes from the coding. The view
    /// writes values and masks in the variable's packed type.
    pub flag_values: Vec<i64>,
    /// `flag_masks` of a bit-flag field.
    pub flag_masks: Vec<i64>,
    /// `flag_meanings`, one per flag value or mask.
    pub flag_meanings: Vec<Box<str>>,
    /// Source attributes with no slot above, verbatim, typed and in file order
    /// (for example CfRadial `grid_mapping`). A per-ray array
    /// ([`AttrValue::ray_alignment`]) moves with its ray in
    /// [`crate::model::Sweep::permute_rays`] and follows the FM301 view's ray
    /// order; any other value stays as stored.
    pub other: Vec<(Box<str>, AttrValue)>,
}

/// One gate's value with its sentinel resolved.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Gate {
    /// A physical value, in the field's units.
    Value(f32),
    /// No data: the fill code, NaN, a value outside `valid_range`, or a row the source did not provide.
    Missing,
    /// Radiated, but no valid echo (NEXRAD code 0, ODIM `undetect`).
    Undetect,
    /// Range folded (NEXRAD code 1).
    RangeFolded,
}

impl Gate {
    /// The physical value of a `Value`, `None` for every sentinel.
    pub fn value(self) -> Option<f32> {
        match self {
            Self::Value(value) => Some(value),
            _ => None,
        }
    }
}

/// A borrowed row of a field in its storage type. Exhaustive, like
/// [`FieldData`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RowRef<'a> {
    /// A row of unsigned 8-bit codes.
    U8(&'a [u8]),
    /// A row of unsigned 16-bit codes.
    U16(&'a [u16]),
    /// A row of signed 8-bit codes.
    I8(&'a [i8]),
    /// A row of signed 16-bit codes.
    I16(&'a [i16]),
    /// A row of signed 32-bit codes.
    I32(&'a [i32]),
    /// A row of 32-bit floats.
    F32(&'a [f32]),
    /// A row of 64-bit floats.
    F64(&'a [f64]),
}

/// A field moved apart into its components (bindings hand `data` to NumPy
/// without copying).
#[derive(Clone, Debug, PartialEq)]
pub struct FieldParts {
    /// See [`Field::name`].
    pub name: FieldName,
    /// See [`Field::quantity`].
    pub quantity: Quantity,
    /// See [`Field::polarization`].
    pub polarization: Polarization,
    /// See [`Field::attrs`].
    pub attrs: FieldAttrs,
    /// See [`Field::nrays`].
    pub nrays: u32,
    /// See [`Field::ngates`].
    pub ngates: u32,
    /// See [`Field::gates`].
    pub gates: GateMapping,
    /// See [`Field::data`].
    pub data: FieldData,
    /// See [`Field::absent_rows`].
    pub absent_rows: Vec<u32>,
}

/// A field's coding, detached from its buffer. Exhaustive, like
/// [`FieldData`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coding {
    /// Coding of unsigned 8-bit codes.
    U8(IntCoding<u8>),
    /// Coding of unsigned 16-bit codes.
    U16(IntCoding<u16>),
    /// Coding of signed 8-bit codes.
    I8(IntCoding<i8>),
    /// Coding of signed 16-bit codes.
    I16(IntCoding<i16>),
    /// Coding of signed 32-bit codes.
    I32(IntCoding<i32>),
    /// Coding of 32-bit floats.
    F32(FloatCoding<f32>),
    /// Coding of 64-bit floats.
    F64(FloatCoding<f64>),
}

/// Error building or filling a [`Field`].
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum FieldError {
    /// A row of one storage type pushed into a field of another.
    #[error("field storage is {expected}, row is {actual}")]
    StorageMismatch {
        /// The field's storage type.
        expected: &'static str,
        /// The row's storage type.
        actual: &'static str,
    },
    /// Rows pushed out of ray order.
    #[error("row for ray {ray} pushed after {rows} rows (rows must be pushed in ray order)")]
    RowOrder {
        /// The ray the row was pushed for.
        ray: usize,
        /// Rows the field already had.
        rows: usize,
    },
    /// A big-endian 16-bit row with an odd number of bytes.
    #[error("big-endian 16-bit row has odd byte length {byte_len}")]
    InvalidRowByteLength {
        /// Length of the row in bytes.
        byte_len: usize,
    },
    /// The field's dimensions do not fit `u32` or addressable memory.
    #[error("field dimensions exceed u32 or addressable memory")]
    TooLarge,
    /// The field has more rows than its sweep has rays.
    #[error("field has {rows} rows, more than the sweep's {nrays} rays")]
    TooManyRows {
        /// Rows of the field.
        rows: usize,
        /// Rays of the sweep.
        nrays: usize,
    },
}

impl FieldData {
    /// Number of stored values (`nrays × ngates`).
    pub fn len(&self) -> usize {
        match self {
            Self::U8 { values, .. } => values.len(),
            Self::U16 { values, .. } => values.len(),
            Self::I8 { values, .. } => values.len(),
            Self::I16 { values, .. } => values.len(),
            Self::I32 { values, .. } => values.len(),
            Self::F32 { values, .. } => values.len(),
            Self::F64 { values, .. } => values.len(),
        }
    }

    /// Whether the field stores no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// NumPy-style dtype name of the storage.
    pub fn dtype(&self) -> &'static str {
        match self {
            Self::U8 { .. } => "uint8",
            Self::U16 { .. } => "uint16",
            Self::I8 { .. } => "int8",
            Self::I16 { .. } => "int16",
            Self::I32 { .. } => "int32",
            Self::F32 { .. } => "float32",
            Self::F64 { .. } => "float64",
        }
    }

    /// The fill code in the storage type (`_FillValue`, else 0 or NaN).
    pub fn fill_scalar(&self) -> Scalar {
        match self {
            Self::U8 { coding, .. } => Scalar::U8(coding.fill_code()),
            Self::U16 { coding, .. } => Scalar::U16(coding.fill_code()),
            Self::I8 { coding, .. } => Scalar::I8(coding.fill_code()),
            Self::I16 { coding, .. } => Scalar::I16(coding.fill_code()),
            Self::I32 { coding, .. } => Scalar::I32(coding.fill_code()),
            Self::F32 { coding, .. } => Scalar::F32(coding.fill_code()),
            Self::F64 { coding, .. } => Scalar::F64(coding.fill_code()),
        }
    }

    /// The coding without the buffer.
    pub fn coding(&self) -> Coding {
        match self {
            Self::U8 { coding, .. } => Coding::U8(*coding),
            Self::U16 { coding, .. } => Coding::U16(*coding),
            Self::I8 { coding, .. } => Coding::I8(*coding),
            Self::I16 { coding, .. } => Coding::I16(*coding),
            Self::I32 { coding, .. } => Coding::I32(*coding),
            Self::F32 { coding, .. } => Coding::F32(*coding),
            Self::F64 { coding, .. } => Coding::F64(*coding),
        }
    }

    /// The linear transform, if the coding has one.
    pub fn transform(&self) -> Option<LinearTransform> {
        match self {
            Self::U8 { coding, .. } => Some(coding.transform),
            Self::U16 { coding, .. } => Some(coding.transform),
            Self::I8 { coding, .. } => Some(coding.transform),
            Self::I16 { coding, .. } => Some(coding.transform),
            Self::I32 { coding, .. } => Some(coding.transform),
            Self::F32 { coding, .. } => coding.transform,
            Self::F64 { coding, .. } => coding.transform,
        }
    }

    /// The buffer, moved, with its coding. No copy.
    pub fn into_array(self) -> (ArrayBuf, Coding) {
        match self {
            Self::U8 { values, coding } => (ArrayBuf::U8(values), Coding::U8(coding)),
            Self::U16 { values, coding } => (ArrayBuf::U16(values), Coding::U16(coding)),
            Self::I8 { values, coding } => (ArrayBuf::I8(values), Coding::I8(coding)),
            Self::I16 { values, coding } => (ArrayBuf::I16(values), Coding::I16(coding)),
            Self::I32 { values, coding } => (ArrayBuf::I32(values), Coding::I32(coding)),
            Self::F32 { values, coding } => (ArrayBuf::F32(values), Coding::F32(coding)),
            Self::F64 { values, coding } => (ArrayBuf::F64(values), Coding::F64(coding)),
        }
    }
}

/// Row bookkeeping shared by the typed push paths.
struct Rows<'a> {
    nrays: &'a mut u32,
    ngates: &'a mut u32,
    absent_rows: &'a mut Vec<u32>,
}

impl Rows<'_> {
    /// Prepare `values` for a row of `row_len` values for ray `ray`: check the
    /// order, widen existing rows if the row is longer, and append absent rows
    /// for skipped rays. Returns the gate count after widening.
    fn prepare<T: Copy>(
        &mut self,
        values: &mut Vec<T>,
        ray: usize,
        row_len: usize,
        fill: T,
    ) -> Result<usize, FieldError> {
        let rows = *self.nrays as usize;
        if ray < rows {
            return Err(FieldError::RowOrder { ray, rows });
        }
        u32::try_from(ray.checked_add(1).ok_or(FieldError::TooLarge)?)
            .map_err(|_| FieldError::TooLarge)?;
        let mut ngates = *self.ngates as usize;
        if row_len > ngates {
            let widened = u32::try_from(row_len).map_err(|_| FieldError::TooLarge)?;
            if rows > 0 && ngates > 0 {
                *values = widen_rows(values, rows, ngates, row_len, fill)?;
            } else if rows > 0 {
                values.clear();
                values.resize(rows.checked_mul(row_len).ok_or(FieldError::TooLarge)?, fill);
            }
            *self.ngates = widened;
            ngates = row_len;
        }
        if ray > rows {
            let extra = (ray - rows)
                .checked_mul(ngates)
                .ok_or(FieldError::TooLarge)?;
            values.reserve(extra);
            values.resize(values.len() + extra, fill);
            self.absent_rows.extend((rows..ray).map(|row| row as u32));
        }
        Ok(ngates)
    }

    fn finish<T: Copy>(
        &mut self,
        values: &mut Vec<T>,
        row_start: usize,
        ngates: usize,
        ray: usize,
        fill: T,
    ) {
        values.resize(row_start + ngates, fill);
        *self.nrays = (ray + 1) as u32;
    }
}

fn widen_rows<T: Copy>(
    values: &[T],
    rows: usize,
    old_gates: usize,
    new_gates: usize,
    fill: T,
) -> Result<Vec<T>, FieldError> {
    let mut widened = Vec::new();
    widened
        .try_reserve_exact(rows.checked_mul(new_gates).ok_or(FieldError::TooLarge)?)
        .map_err(|_| FieldError::TooLarge)?;
    for row in values.chunks(old_gates).take(rows) {
        widened.extend_from_slice(row);
        widened.resize(widened.len() + (new_gates - row.len()), fill);
    }
    Ok(widened)
}

macro_rules! push_slice_row {
    ($(#[$doc:meta])* $fn_name:ident, $variant:ident, $t:ty) => {
        $(#[$doc])*
        pub fn $fn_name(&mut self, ray: usize, row: &[$t]) -> Result<(), FieldError> {
            let Field { data, nrays, ngates, absent_rows, .. } = self;
            let FieldData::$variant { values, coding } = data else {
                return Err(FieldError::StorageMismatch {
                    expected: data.dtype(),
                    actual: stringify!($t),
                });
            };
            let fill = coding.fill_code();
            let mut rows = Rows { nrays, ngates, absent_rows };
            let ngates = rows.prepare(values, ray, row.len(), fill)?;
            let row_start = values.len();
            values.extend_from_slice(row);
            rows.finish(values, row_start, ngates, ray, fill);
            Ok(())
        }
    };
}

macro_rules! push_be_row {
    ($(#[$doc:meta])* $fn_name:ident, $variant:ident, $t:ty) => {
        $(#[$doc])*
        pub fn $fn_name(&mut self, ray: usize, row_be: &[u8]) -> Result<(), FieldError> {
            if !row_be.len().is_multiple_of(2) {
                return Err(FieldError::InvalidRowByteLength { byte_len: row_be.len() });
            }
            let Field { data, nrays, ngates, absent_rows, .. } = self;
            let FieldData::$variant { values, coding } = data else {
                return Err(FieldError::StorageMismatch {
                    expected: data.dtype(),
                    actual: stringify!($t),
                });
            };
            let fill = coding.fill_code();
            let mut rows = Rows { nrays, ngates, absent_rows };
            let ngates = rows.prepare(values, ray, row_be.len() / 2, fill)?;
            let row_start = values.len();
            values.extend(
                row_be
                    .chunks_exact(2)
                    .map(|pair| <$t>::from_be_bytes([pair[0], pair[1]])),
            );
            rows.finish(values, row_start, ngates, ray, fill);
            Ok(())
        }
    };
}

impl Field {
    /// A field over existing (possibly empty) row-major data. `nrays` is
    /// `data.len() / ngates`; `quantity` and `polarization` come from
    /// [`Quantity::classify`] on the name; attributes start empty.
    pub fn new(name: FieldName, gates: GateMapping, ngates: u32, data: FieldData) -> Self {
        let (quantity, polarization) = Quantity::classify(name.as_str(), None);
        let nrays = if ngates == 0 {
            0
        } else {
            u32::try_from(data.len() / ngates as usize).unwrap_or(u32::MAX)
        };
        Self {
            name,
            quantity,
            polarization,
            attrs: FieldAttrs::default(),
            nrays,
            ngates,
            gates,
            data,
            absent_rows: Vec::new(),
        }
    }

    /// `(nrays, ngates)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrays as usize, self.ngates as usize)
    }

    /// `true` when the source did not provide row `ray`.
    #[inline]
    pub fn is_absent(&self, ray: usize) -> bool {
        !self.absent_rows.is_empty()
            && u32::try_from(ray).is_ok_and(|ray| self.absent_rows.binary_search(&ray).is_ok())
    }

    /// Row `ray` in its storage type.
    pub fn row(&self, ray: usize) -> Option<RowRef<'_>> {
        let ngates = self.ngates as usize;
        if ray >= self.nrays as usize {
            return None;
        }
        let start = ray.checked_mul(ngates)?;
        let end = start.checked_add(ngates)?;
        Some(match &self.data {
            FieldData::U8 { values, .. } => RowRef::U8(values.get(start..end)?),
            FieldData::U16 { values, .. } => RowRef::U16(values.get(start..end)?),
            FieldData::I8 { values, .. } => RowRef::I8(values.get(start..end)?),
            FieldData::I16 { values, .. } => RowRef::I16(values.get(start..end)?),
            FieldData::I32 { values, .. } => RowRef::I32(values.get(start..end)?),
            FieldData::F32 { values, .. } => RowRef::F32(values.get(start..end)?),
            FieldData::F64 { values, .. } => RowRef::F64(values.get(start..end)?),
        })
    }

    /// Resolve a native gate, in this order:
    /// 1. `ray` is in `absent_rows`: `Missing`.
    /// 2. The raw value equals `undetect`: `Undetect` (so NEXRAD raw 0 reads as
    ///    `Undetect` in a provided row).
    /// 3. It equals `fill_value` or is NaN: `Missing`.
    /// 4. It equals `range_folded`: `RangeFolded`.
    /// 5. It is outside `valid_range`: `Missing`.
    /// 6. Otherwise: `Value(physical)`.
    ///
    /// Gates past the native extent are not native gates (`None`).
    pub fn gate(&self, ray: usize, gate: usize) -> Option<Gate> {
        let ngates = self.ngates as usize;
        if ray >= self.nrays as usize || gate >= ngates {
            return None;
        }
        if self.is_absent(ray) {
            return Some(Gate::Missing);
        }
        let index = ray.checked_mul(ngates)?.checked_add(gate)?;
        Some(match &self.data {
            FieldData::U8 { values, coding } => coding.resolve(*values.get(index)?),
            FieldData::U16 { values, coding } => coding.resolve(*values.get(index)?),
            FieldData::I8 { values, coding } => coding.resolve(*values.get(index)?),
            FieldData::I16 { values, coding } => coding.resolve(*values.get(index)?),
            FieldData::I32 { values, coding } => coding.resolve(*values.get(index)?),
            FieldData::F32 { values, coding } => coding.resolve(*values.get(index)?),
            FieldData::F64 { values, coding } => coding.resolve(*values.get(index)?),
        })
    }

    /// Physical value of a native gate; `None` for every sentinel and outside
    /// the native extent.
    #[inline]
    pub fn value(&self, ray: usize, gate: usize) -> Option<f32> {
        self.gate(ray, gate)?.value()
    }

    /// 256-entry decode table for `u8` / `i8` fields, indexed by the raw byte
    /// (`raw as u8` for `i8`), NaN for every sentinel. Absent rows hold the fill
    /// code, so they decode to NaN only when the coding has a `fill_value`;
    /// check [`Field::is_absent`] otherwise.
    pub fn lut8(&self) -> Option<[f32; 256]> {
        let mut table = [f32::NAN; 256];
        match &self.data {
            FieldData::U8 { coding, .. } => {
                for (code, slot) in table.iter_mut().enumerate() {
                    *slot = coding.resolve(code as u8).value().unwrap_or(f32::NAN);
                }
            }
            FieldData::I8 { coding, .. } => {
                for (code, slot) in table.iter_mut().enumerate() {
                    *slot = coding.resolve(code as u8 as i8).value().unwrap_or(f32::NAN);
                }
            }
            _ => return None,
        }
        Some(table)
    }

    /// Explicit float expansion, row-major `[nrays × ngates]`, NaN for every
    /// sentinel. Decoders never call this.
    ///
    /// Every value is [`Field::value`] of its gate (NaN where that is `None`),
    /// computed per code rather than per gate: 8-bit codes and large 16-bit
    /// fields go through a decode table built with the coding's own
    /// `resolve`, so the result is bit for bit the per-gate one.
    pub fn to_physical(&self) -> Vec<f32> {
        let (nrays, ngates) = self.shape();
        let len = nrays.saturating_mul(ngates);
        let resolved = |gate: Gate| gate.value().unwrap_or(f32::NAN);
        let mut out: Vec<f32> = match &self.data {
            FieldData::U8 { values, coding } => {
                let table: [f32; 256] =
                    std::array::from_fn(|code| resolved(coding.resolve(code as u8)));
                values
                    .iter()
                    .take(len)
                    .map(|raw| table[usize::from(*raw)])
                    .collect()
            }
            FieldData::I8 { values, coding } => {
                let table: [f32; 256] =
                    std::array::from_fn(|code| resolved(coding.resolve(code as u8 as i8)));
                values
                    .iter()
                    .take(len)
                    .map(|raw| table[usize::from(*raw as u8)])
                    .collect()
            }
            FieldData::U16 { values, coding } => {
                let values = &values[..values.len().min(len)];
                if values.len() >= LUT16_MIN_VALUES {
                    let table: Vec<f32> = (0..=u16::MAX)
                        .map(|code| resolved(coding.resolve(code)))
                        .collect();
                    values.iter().map(|raw| table[usize::from(*raw)]).collect()
                } else {
                    values
                        .iter()
                        .map(|raw| resolved(coding.resolve(*raw)))
                        .collect()
                }
            }
            FieldData::I16 { values, coding } => {
                let values = &values[..values.len().min(len)];
                if values.len() >= LUT16_MIN_VALUES {
                    let table: Vec<f32> = (0..=u16::MAX)
                        .map(|code| resolved(coding.resolve(code as i16)))
                        .collect();
                    values
                        .iter()
                        .map(|raw| table[usize::from(*raw as u16)])
                        .collect()
                } else {
                    values
                        .iter()
                        .map(|raw| resolved(coding.resolve(*raw)))
                        .collect()
                }
            }
            FieldData::I32 { values, coding } => values
                .iter()
                .take(len)
                .map(|raw| resolved(coding.resolve(*raw)))
                .collect(),
            FieldData::F32 { values, coding } => values
                .iter()
                .take(len)
                .map(|raw| resolved(coding.resolve(*raw)))
                .collect(),
            FieldData::F64 { values, coding } => values
                .iter()
                .take(len)
                .map(|raw| resolved(coding.resolve(*raw)))
                .collect(),
        };
        // Gates past the stored values have no value.
        out.resize(len, f32::NAN);
        for &row in &self.absent_rows {
            let start = (row as usize).saturating_mul(ngates);
            if let Some(slots) = out.get_mut(start..start.saturating_add(ngates)) {
                slots.fill(f32::NAN);
            }
        }
        out
    }

    /// Native geometry on `range`: (centre of native gate 0 in metres, native
    /// spacing in metres). For an explicit range the spacing is the distance to
    /// the next centre (0 for a single gate).
    pub fn native_geometry(&self, range: &RangeCoord) -> Option<(f64, f64)> {
        match range {
            RangeCoord::Uniform {
                first_center_m,
                spacing_m,
                ..
            } => {
                let stride = f64::from(self.gates.stride.max(1));
                let start_edge =
                    first_center_m - spacing_m / 2.0 + f64::from(self.gates.start) * spacing_m;
                let spacing = spacing_m * stride;
                Some((start_edge + spacing / 2.0, spacing))
            }
            RangeCoord::Explicit { centers_m } => {
                let start = self.gates.start as usize;
                let center = f64::from(*centers_m.get(start)?);
                let spacing = centers_m
                    .get(start + 1)
                    .map_or(0.0, |next| f64::from(*next) - center);
                Some((center, spacing))
            }
        }
    }

    /// Reserve storage for exactly `rows` more rows of `ngates` values
    /// (`Vec::reserve_exact`: no doubling, so a decoder that learns the row
    /// count late can grow a field by a small step).
    pub fn reserve_rows(&mut self, rows: usize) {
        let additional = rows.saturating_mul(self.ngates as usize);
        match &mut self.data {
            FieldData::U8 { values, .. } => values.reserve_exact(additional),
            FieldData::U16 { values, .. } => values.reserve_exact(additional),
            FieldData::I8 { values, .. } => values.reserve_exact(additional),
            FieldData::I16 { values, .. } => values.reserve_exact(additional),
            FieldData::I32 { values, .. } => values.reserve_exact(additional),
            FieldData::F32 { values, .. } => values.reserve_exact(additional),
            FieldData::F64 { values, .. } => values.reserve_exact(additional),
        }
    }

    push_slice_row!(
        /// Push the `u8` row of ray `ray` (the index `Sweep::push_ray` returned).
        /// Rays skipped since the last row become absent rows; a short row is
        /// padded with the fill code; a longer row widens every existing row.
        push_row_u8, U8, u8
    );
    push_slice_row!(
        /// Push a `u16` row; see [`Field::push_row_u8`].
        push_row_u16, U16, u16
    );
    push_be_row!(
        /// Push a big-endian `u16` row (NEXRAD 16-bit moments); see
        /// [`Field::push_row_u8`].
        push_row_u16_be, U16, u16
    );
    push_slice_row!(
        /// Push an `i8` row; see [`Field::push_row_u8`].
        push_row_i8, I8, i8
    );
    push_slice_row!(
        /// Push an `i16` row (DORADE 16-bit fields); see
        /// [`Field::push_row_u8`].
        push_row_i16, I16, i16
    );
    push_be_row!(
        /// Push a big-endian `i16` row; see [`Field::push_row_u8`].
        push_row_i16_be, I16, i16
    );
    push_slice_row!(
        /// Push an `i32` row (CfRadial `int` fields); see
        /// [`Field::push_row_u8`].
        push_row_i32, I32, i32
    );
    push_slice_row!(
        /// Push an `f32` row; see [`Field::push_row_u8`].
        push_row_f32, F32, f32
    );
    push_slice_row!(
        /// Push an `f64` row; see [`Field::push_row_u8`].
        push_row_f64, F64, f64
    );

    /// Append absent rows until the field has `rays` rows (`Sweep::seal` calls
    /// this for rays at the end that the field never received).
    pub fn push_absent_rows_to(&mut self, rays: usize) -> Result<(), FieldError> {
        let rows = self.nrays as usize;
        if rows > rays {
            return Err(FieldError::TooManyRows { rows, nrays: rays });
        }
        if rows == rays {
            return Ok(());
        }
        let Field {
            data,
            nrays,
            ngates,
            absent_rows,
            ..
        } = self;
        let mut state = Rows {
            nrays,
            ngates,
            absent_rows,
        };
        macro_rules! fill_to {
            ($values:expr, $fill:expr) => {{
                let values = $values;
                let fill = $fill;
                // Rows `rows..rays-1` are absent; the last one is pushed as an
                // absent row too.
                let ngates = state.prepare(values, rays - 1, 0, fill)?;
                let row_start = values.len();
                state.finish(values, row_start, ngates, rays - 1, fill);
                state.absent_rows.push((rays - 1) as u32);
            }};
        }
        match data {
            FieldData::U8 { values, coding } => fill_to!(values, coding.fill_code()),
            FieldData::U16 { values, coding } => fill_to!(values, coding.fill_code()),
            FieldData::I8 { values, coding } => fill_to!(values, coding.fill_code()),
            FieldData::I16 { values, coding } => fill_to!(values, coding.fill_code()),
            FieldData::I32 { values, coding } => fill_to!(values, coding.fill_code()),
            FieldData::F32 { values, coding } => fill_to!(values, coding.fill_code()),
            FieldData::F64 { values, coding } => fill_to!(values, coding.fill_code()),
        }
        Ok(())
    }

    /// Move the field apart. No copy.
    pub fn into_parts(self) -> FieldParts {
        FieldParts {
            name: self.name,
            quantity: self.quantity,
            polarization: self.polarization,
            attrs: self.attrs,
            nrays: self.nrays,
            ngates: self.ngates,
            gates: self.gates,
            data: self.data,
            absent_rows: self.absent_rows,
        }
    }
}
