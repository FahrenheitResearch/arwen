//! Typed scalar, array and attribute values shared by the model and the FM301
//! view (`docs/design/fm301-model.md` section 2).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A numeric scalar that keeps its type.
///
/// CF requires `_FillValue`, `valid_range` and `flag_values` in the packed
/// variable's type, and xarray picks the decoded dtype from the type of
/// `scale_factor` (float32 attributes on 8/16-bit data decode to float32).
///
/// Equality is structural: floats compare by bit pattern ([`Scalar::bit_eq`]),
/// so a NaN a source file stores equals itself and two decodes of the same
/// bytes compare equal.
///
/// Not `#[non_exhaustive]`, on purpose: the variants are the numeric types
/// of the data model, and writers and bindings that convert values must
/// handle every one (see [`crate::model::FieldData`]). [`ArrayBuf`] is
/// exhaustive for the same reason.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Scalar {
    /// A signed 8-bit value.
    I8(i8),
    /// An unsigned 8-bit value.
    U8(u8),
    /// A signed 16-bit value.
    I16(i16),
    /// An unsigned 16-bit value.
    U16(u16),
    /// A signed 32-bit value.
    I32(i32),
    /// An unsigned 32-bit value.
    U32(u32),
    /// A signed 64-bit value.
    I64(i64),
    /// An unsigned 64-bit value.
    U64(u64),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit float.
    F64(f64),
}

impl Scalar {
    /// The value widened to `f64` (exact for every integer type up to 32 bits).
    pub fn as_f64(self) -> f64 {
        match self {
            Self::I8(v) => f64::from(v),
            Self::U8(v) => f64::from(v),
            Self::I16(v) => f64::from(v),
            Self::U16(v) => f64::from(v),
            Self::I32(v) => f64::from(v),
            Self::U32(v) => f64::from(v),
            Self::I64(v) => v as f64,
            Self::U64(v) => v as f64,
            Self::F32(v) => f64::from(v),
            Self::F64(v) => v,
        }
    }

    /// NumPy-style dtype name (`"uint8"`, `"float64"`, ...).
    pub fn dtype(self) -> &'static str {
        match self {
            Self::I8(_) => "int8",
            Self::U8(_) => "uint8",
            Self::I16(_) => "int16",
            Self::U16(_) => "uint16",
            Self::I32(_) => "int32",
            Self::U32(_) => "uint32",
            Self::I64(_) => "int64",
            Self::U64(_) => "uint64",
            Self::F32(_) => "float32",
            Self::F64(_) => "float64",
        }
    }

    /// Structural equality with floats compared by bit pattern (NaN equals NaN).
    pub fn bit_eq(self, other: Self) -> bool {
        match (self, other) {
            (Self::F32(a), Self::F32(b)) => a.to_bits() == b.to_bits(),
            (Self::F64(a), Self::F64(b)) => a.to_bits() == b.to_bits(),
            (Self::I8(a), Self::I8(b)) => a == b,
            (Self::U8(a), Self::U8(b)) => a == b,
            (Self::I16(a), Self::I16(b)) => a == b,
            (Self::U16(a), Self::U16(b)) => a == b,
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::I64(a), Self::I64(b)) => a == b,
            (Self::U64(a), Self::U64(b)) => a == b,
            _ => false,
        }
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.bit_eq(*other)
    }
}

/// A typed 1-D buffer, row-major when its owner has more than one dimension.
///
/// Equality is structural: float elements compare by bit pattern, as for
/// [`Scalar`], so buffers holding a source's NaN values compare equal to
/// themselves. Exhaustive, like [`Scalar`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ArrayBuf {
    /// Signed 8-bit values.
    I8(Vec<i8>),
    /// Unsigned 8-bit values.
    U8(Vec<u8>),
    /// Signed 16-bit values.
    I16(Vec<i16>),
    /// Unsigned 16-bit values.
    U16(Vec<u16>),
    /// Signed 32-bit values.
    I32(Vec<i32>),
    /// Unsigned 32-bit values.
    U32(Vec<u32>),
    /// Signed 64-bit values.
    I64(Vec<i64>),
    /// 32-bit floats.
    F32(Vec<f32>),
    /// 64-bit floats.
    F64(Vec<f64>),
    /// Strings (netCDF `string` or `char` arrays).
    Text(Vec<Box<str>>),
}

impl PartialEq for ArrayBuf {
    fn eq(&self, other: &Self) -> bool {
        fn bits<T: Copy, B: PartialEq>(a: &[T], b: &[T], to_bits: fn(T) -> B) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| to_bits(*x) == to_bits(*y))
        }
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => a == b,
            (Self::U8(a), Self::U8(b)) => a == b,
            (Self::I16(a), Self::I16(b)) => a == b,
            (Self::U16(a), Self::U16(b)) => a == b,
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::I64(a), Self::I64(b)) => a == b,
            (Self::F32(a), Self::F32(b)) => bits(a, b, f32::to_bits),
            (Self::F64(a), Self::F64(b)) => bits(a, b, f64::to_bits),
            (Self::Text(a), Self::Text(b)) => a == b,
            _ => false,
        }
    }
}

impl ArrayBuf {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::I8(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Text(v) => v.len(),
        }
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// NumPy-style dtype name; `"str"` for text.
    pub fn dtype(&self) -> &'static str {
        match self {
            Self::I8(_) => "int8",
            Self::U8(_) => "uint8",
            Self::I16(_) => "int16",
            Self::U16(_) => "uint16",
            Self::I32(_) => "int32",
            Self::U32(_) => "uint32",
            Self::I64(_) => "int64",
            Self::F32(_) => "float32",
            Self::F64(_) => "float64",
            Self::Text(_) => "str",
        }
    }

    /// Element `index` as `f64`, or `None` for text or out of range.
    pub fn get_f64(&self, index: usize) -> Option<f64> {
        Some(match self {
            Self::I8(v) => f64::from(*v.get(index)?),
            Self::U8(v) => f64::from(*v.get(index)?),
            Self::I16(v) => f64::from(*v.get(index)?),
            Self::U16(v) => f64::from(*v.get(index)?),
            Self::I32(v) => f64::from(*v.get(index)?),
            Self::U32(v) => f64::from(*v.get(index)?),
            Self::I64(v) => *v.get(index)? as f64,
            Self::F32(v) => f64::from(*v.get(index)?),
            Self::F64(v) => *v.get(index)?,
            Self::Text(_) => return None,
        })
    }

    /// Rows of `self` (row-major, `row_len` elements each) taken in `order`.
    /// Returns `None` when `order` indexes past the end.
    ///
    /// Every row is checked before the output is reserved, so a `row_len`
    /// larger than `self` returns `None` without allocating.
    pub fn take_rows(&self, row_len: usize, order: &[u32]) -> Option<Self> {
        fn take<T: Clone>(values: &[T], row_len: usize, order: &[u32]) -> Option<Vec<T>> {
            // Rows that exist; with `row_len` 0 every row is empty.
            let rows = values.len().checked_div(row_len).unwrap_or(usize::MAX);
            if order.iter().any(|&row| row as usize >= rows) {
                return None;
            }
            let mut out = Vec::with_capacity(order.len().saturating_mul(row_len));
            for &row in order {
                let start = (row as usize).checked_mul(row_len)?;
                out.extend_from_slice(values.get(start..start.checked_add(row_len)?)?);
            }
            Some(out)
        }
        Some(match self {
            Self::I8(v) => Self::I8(take(v, row_len, order)?),
            Self::U8(v) => Self::U8(take(v, row_len, order)?),
            Self::I16(v) => Self::I16(take(v, row_len, order)?),
            Self::U16(v) => Self::U16(take(v, row_len, order)?),
            Self::I32(v) => Self::I32(take(v, row_len, order)?),
            Self::U32(v) => Self::U32(take(v, row_len, order)?),
            Self::I64(v) => Self::I64(take(v, row_len, order)?),
            Self::F32(v) => Self::F32(take(v, row_len, order)?),
            Self::F64(v) => Self::F64(take(v, row_len, order)?),
            Self::Text(v) => Self::Text(take(v, row_len, order)?),
        })
    }
}

/// An attribute value with its type.
///
/// `Bool` exists because xradar 0.12 writes Python bools for NEXRAD attributes
/// (`mpda_vcp: False`). netCDF has no bool type, so the WMO flavor and file
/// writers emit `"true"`/`"false"` text, FM301's convention.
///
/// Exhaustive, like [`Scalar`]: the file writers write every attribute back
/// in its type, so a new kind of value is a breaking change instead of a case
/// a wildcard arm would silently drop.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AttrValue {
    /// A text attribute.
    Text(Box<str>),
    /// A boolean attribute (xradar writes some NEXRAD attributes as Python booleans).
    Bool(bool),
    /// A numeric attribute of one value, in its source type.
    Scalar(Scalar),
    /// A numeric or string attribute of several values.
    Array(ArrayBuf),
}

impl AttrValue {
    /// A text attribute.
    pub fn text(value: impl Into<Box<str>>) -> Self {
        Self::Text(value.into())
    }

    /// The text of a `Text` value.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            _ => None,
        }
    }

    /// A numeric scalar (or a bool as 0/1) widened to `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Scalar(scalar) => Some(scalar.as_f64()),
            Self::Bool(value) => Some(f64::from(u8::from(*value))),
            Self::Array(array) if array.len() == 1 => array.get_f64(0),
            _ => None,
        }
    }

    /// How this value, a verbatim attribute named `name` of a sweep of
    /// `nrays` rays or of one of its fields ([`crate::model::Sweep::other`],
    /// [`crate::model::FieldAttrs::other`]), lines up with the rays.
    ///
    /// Verbatim attributes are untyped, so length alone cannot tell a
    /// per-ray array from another array that happens to have `nrays`
    /// entries. An array of `nrays` entries (more than two rays: a
    /// two-element attribute is far more often a range, CF `valid_range`)
    /// is [`RayAlignment::PerRay`] when `name` is one of the per-ray `how`
    /// arrays of ODIM_H5 (`startazA`, `stopazA`, `startazT`, `stopazT`,
    /// `startelA`, `stopelA`, `startelT`, `stopelT`, `elangles`, `TXpower`)
    /// or of a national ODIM feed (Bureau of Meteorology: `dataflag`,
    /// `noisepowerh`, `noisepowerv`, `numpulses`, `startT`), and
    /// [`RayAlignment::Unknown`] otherwise. The list is not part of the API:
    /// it goes when io-odim gives these arrays a `time` dimension in
    /// [`crate::model::Sweep::extra_vars`].
    pub fn ray_alignment(&self, name: &str, nrays: usize) -> RayAlignment {
        match self {
            Self::Array(array) if nrays > 2 && array.len() == nrays => {
                if PER_RAY_ATTRIBUTES.contains(&name) {
                    RayAlignment::PerRay
                } else {
                    RayAlignment::Unknown
                }
            }
            _ => RayAlignment::NotPerRay,
        }
    }

    /// This value with an array of `nrays` entries taken in `order` (entry
    /// `i` is entry `order[i]` before); any other value, or an `order` that
    /// indexes past the array, unchanged. The caller decides that the array
    /// is per ray ([`AttrValue::ray_alignment`]).
    pub fn in_ray_order(&self, nrays: usize, order: &[u32]) -> AttrValue {
        match self {
            Self::Array(array) if array.len() == nrays && order.len() == nrays => array
                .take_rows(1, order)
                .map_or_else(|| self.clone(), Self::Array),
            _ => self.clone(),
        }
    }
}

/// Names of the verbatim attribute arrays that hold one value per ray when
/// they have one entry per ray of their sweep (`AttrValue::ray_alignment`):
/// the ODIM_H5 per-ray `how` arrays (`startazA`, `stopazA`, `startazT`,
/// `stopazT`, `startelA`, `stopelA`, `startelT`, `stopelT`, `elangles`,
/// `TXpower`) and the ones national ODIM feeds add (Bureau of Meteorology:
/// `dataflag`, `noisepowerh`, `noisepowerv`, `numpulses`, `startT`). In 130
/// real ODIM files (the committed fixtures and the feed corpus) every `how`
/// array with one entry per ray has one of these names, and every other
/// `how` array (`key_values`, `resolution`) has another length. Sorted.
pub(crate) const PER_RAY_ATTRIBUTES: &[&str] = &[
    "TXpower",
    "dataflag",
    "elangles",
    "noisepowerh",
    "noisepowerv",
    "numpulses",
    "startT",
    "startazA",
    "startazT",
    "startelA",
    "startelT",
    "stopazA",
    "stopazT",
    "stopelA",
    "stopelT",
];

/// How a verbatim attribute lines up with the rays of its sweep
/// ([`AttrValue::ray_alignment`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RayAlignment {
    /// Not an array with one entry per ray: kept as stored everywhere.
    NotPerRay,
    /// A per-ray array (one of the names [`AttrValue::ray_alignment`]
    /// lists): it moves with its rays in
    /// [`crate::model::Sweep::permute_rays`], and the FM301 view writes it
    /// in the view's ray order, so that entry `i` belongs to the view's ray
    /// `i`.
    PerRay,
    /// An array with one entry per ray under a name not known to be per
    /// ray. Moving it with the rays could scramble an array that is not per
    /// ray, and leaving it could misalign one that is, so it is kept as
    /// stored (the view writes it verbatim, in source order) and
    /// [`crate::model::Sweep::permute_rays`] refuses to reorder its sweep.
    /// A decoder that knows such an array is per ray puts it in
    /// [`crate::model::Sweep::extra_vars`] with a `time` dimension instead.
    Unknown,
}

impl From<Scalar> for AttrValue {
    fn from(value: Scalar) -> Self {
        Self::Scalar(value)
    }
}

impl From<&str> for AttrValue {
    fn from(value: &str) -> Self {
        Self::Text(value.into())
    }
}

impl From<String> for AttrValue {
    fn from(value: String) -> Self {
        Self::Text(value.into_boxed_str())
    }
}

/// A source variable without a typed slot, kept verbatim with its name, type
/// and attributes.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExtraVariable {
    /// Source name (CfRadial `georef_time`, `ray_start_range`, `status_xml`).
    pub name: Box<str>,
    /// Dimension names: `[]`, `["time"]`, `["time", "<source dim>"]`. A `"time"`
    /// dimension has `nrays` entries and follows the view's ray order.
    pub dims: Vec<Box<str>>,
    /// Length of each dimension, in `dims` order: one entry per dimension
    /// name, and their product is the number of values (`[]` for one
    /// value). Deserialization checks both.
    pub shape: Vec<u32>,
    /// The values, row-major.
    pub values: ArrayBuf,
    /// The variable's attributes, typed and in source order.
    pub attrs: Vec<(Box<str>, AttrValue)>,
}

impl ExtraVariable {
    /// `true` when the first dimension is the ray (`"time"`) dimension.
    pub fn is_per_ray(&self) -> bool {
        self.dims.first().is_some_and(|dim| &**dim == "time")
    }
}

/// The source's own attributes of one variable whose values the model keeps
/// in a typed slot (a coordinate, an instrument or calibration variable),
/// verbatim.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VariableAttrs {
    /// The FM301 group the variable belongs to: `""` for the root,
    /// `sweep_<index>` (the model's sweep index), `sweep_<index>/monitoring`,
    /// `radar_parameters`, `radar_calibration`, `georeferencing_correction`.
    /// A CfRadial 1 file keeps every variable at the root.
    pub group: Box<str>,
    /// The variable's name in the source.
    pub name: Box<str>,
    /// Its attributes, typed and in file order.
    pub attrs: Vec<(Box<str>, AttrValue)>,
}
