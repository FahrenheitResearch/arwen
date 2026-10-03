//! The FM301 group view over a [`Volume`]: the conformance surface for xradar
//! `DataTree` output and a future CfRadial 2 writer
//! (`docs/design/fm301-model.md` section 12).
//!
//! [`volume_view`] builds the FM301 group tree (`/`, `radar_parameters`,
//! `radar_calibration`, `georeferencing_correction`, `sweep_<n>`, `monitoring`)
//! with names, dimensions and typed attributes, borrowing field storage.
//! Fields are always written encoded (packed integers with CF attributes, or
//! floats with their `_FillValue`); decoding is the consumer's job. Ray order
//! and gate padding / repetition are described, not applied: a variable whose
//! storage is not already the FM301 array is [`Values::Mapped`], and
//! [`Values::materialize`] applies the mapping on request.
//!
//! [`VolumeView::layout`] detaches the same tree from the volume, so a binding
//! can record the layout and then move each field's buffer out of the volume.

mod build;
mod layout;

use std::borrow::Cow;
use std::sync::Arc;

use thiserror::Error;

pub use build::volume_view;
pub use layout::{DataRef, GroupLayout, VariableLayout, VolumeLayout};

use crate::model::{ArrayBuf, AttrValue, GateMapping, Scalar, SweepError, Volume};

/// Which reference the view reproduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Flavor {
    /// xradar 0.12 names, attribute strings and attribute types
    /// (`sweep_fixed_angle`, "not_set", "meters per seconds", bool
    /// attributes). Used by conformance tests and the DataTree binding.
    Xradar012,
    /// The FM301-2022 text: `fixed_angle`, `Conventions`, `wmo__cf_profile`,
    /// UDUNITS units, Table 301-12a names. Used by a future CfRadial 2 writer.
    Wmo2022,
}

/// Ray dimension and ray order. Only a row permutation changes; no data moves.
///
/// A field whose storage order already is the view's ray order is
/// [`Values::Borrowed`] (in a layout, [`DataRef::is_zero_copy`]); otherwise the
/// view carries a [`RowOrder::Permutation`]. Storage keeps the source's order,
/// so which choice avoids the permutation depends on the format (design note
/// 12.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FirstDim {
    /// Dimension `time`; rays in acquisition order (stable sort by `time_s`).
    /// xradar's `first_dim="time"`, and the only choice for
    /// [`Flavor::Wmo2022`] (a CF coordinate must be monotonic). The permutation
    /// is the identity when storage order is acquisition order (NEXRAD,
    /// CfRadial, most DORADE files) or every ray time is equal (JMA, ODIM
    /// without per-ray times). It is not for ODIM with per-ray times (azimuth
    /// order, starting where the scan did not) or for DORADE files whose ray
    /// times run backwards (NOXP 2009).
    Time,
    /// xradar's default `first_dim="auto"`: dimension `azimuth` (or `elevation`
    /// for RHI sweeps of non-CfRadial sources, as observed in xradar 0.12), rays
    /// sorted by that angle. The permutation is the identity for azimuth-ordered
    /// storage (ODIM, NOXP 2009), and not for NEXRAD, CfRadial or JMA, whose
    /// sweeps start at an arbitrary azimuth.
    Auto,
}

/// Which unmodelled source items the view writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Passthrough {
    /// What the flavor's reference writes. Xradar012: sweep `extra_vars`,
    /// platform track, calibration `extra` and field `attrs.other` yes
    /// (xradar keeps them for CfRadial); root `attrs.other`, sweep `other`
    /// and the field `attrs.other` of an ODIM source no (xradar 0.12 writes
    /// no ODIM `what`/`how` extras or DORADE descriptor attributes).
    /// Wmo2022: FM301 names only.
    Flavor,
    /// Also every `other`, `extra_vars` and `extra` item, verbatim, for lossless
    /// CfRadial 2 output (the global attributes of a CfRadial file, ODIM
    /// `how` attributes, DORADE VOLD text). Verbatim except for ray order: a
    /// per-ray attribute array ([`crate::model::AttrValue::ray_alignment`],
    /// an ODIM `how/TXpower`) is written in the view's ray order, like the
    /// per-ray variables, so that its entries line up with the ray
    /// dimension; the same holds for a field's `other` attributes, which the
    /// Xradar flavor writes. Every other attribute, an array of one entry
    /// per ray under a name not known to be per ray included, is written as
    /// stored.
    All,
}

/// How [`volume_view`] names, orders and filters what it presents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewOptions {
    /// Whose names and attribute conventions the view reproduces.
    pub flavor: Flavor,
    /// The ray dimension and ray order.
    pub first_dim: FirstDim,
    /// Which unmodelled source items the view writes.
    pub passthrough: Passthrough,
}

impl ViewOptions {
    /// xradar 0.12 defaults: `Xradar012`, `first_dim="auto"`.
    pub const XRADAR: ViewOptions = ViewOptions {
        flavor: Flavor::Xradar012,
        first_dim: FirstDim::Auto,
        passthrough: Passthrough::Flavor,
    };
    /// FM301-2022 text, acquisition order.
    pub const WMO: ViewOptions = ViewOptions {
        flavor: Flavor::Wmo2022,
        first_dim: FirstDim::Time,
        passthrough: Passthrough::Flavor,
    };
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self::XRADAR
    }
}

/// The FM301 group tree of one volume.
pub struct VolumeView<'a> {
    /// `/` with children `radar_parameters`, `radar_calibration`,
    /// `georeferencing_correction` (when present) and `sweep_<n>`. A sweep
    /// group's child is `monitoring` (Table 301-11).
    pub root: Group<'a>,
    /// Non-fatal conditions found while building the view.
    pub warnings: Vec<ViewWarning>,
    volume: &'a Volume,
}

/// One group: dimensions, variables, attributes and child groups.
#[derive(Clone, Debug, PartialEq)]
pub struct Group<'a> {
    /// `""` for the root, `"sweep_0"`, `"monitoring"`, ...
    pub name: Cow<'a, str>,
    /// `("time", 720)`, `("range", 1832)`, `("frequency", 1)`.
    pub dims: Vec<(Cow<'a, str>, usize)>,
    /// The group's variables, coordinates first.
    pub variables: Vec<Variable<'a>>,
    /// The group's attributes, typed.
    pub attrs: Vec<(Cow<'a, str>, AttrValue)>,
    /// Child groups (`radar_parameters`, `sweep_0`, `monitoring`, ...).
    pub children: Vec<Group<'a>>,
}

/// One variable in its encoded form.
#[derive(Clone, Debug, PartialEq)]
pub struct Variable<'a> {
    /// Variable name (`DBZH`, `time`, `sweep_fixed_angle`, ...).
    pub name: Cow<'a, str>,
    /// Dimension names, outermost first.
    pub dims: Vec<Cow<'a, str>>,
    /// The variable's values in their encoded form.
    pub values: Values<'a>,
    /// Attributes of the encoded form, typed. Packing, fill and flag attributes
    /// are in the variable's packed type.
    pub attrs: Vec<(Cow<'a, str>, AttrValue)>,
    /// The model field this variable reads, for dataset variables.
    pub source: Option<FieldSource>,
}

/// Variable values.
///
/// Exhaustive, like [`crate::model::FieldData`]: the file writers write
/// every form of values, so a new form is a breaking change instead of a
/// case a wildcard arm would silently mishandle.
#[derive(Clone, Debug, PartialEq)]
pub enum Values<'a> {
    /// Contiguous, same shape and ray order as the variable: zero-copy.
    Borrowed(ArrayRef<'a>),
    /// A field read through its mapping: rows taken in `rows` order, native
    /// gates padded with `fill` and repeated `mapping.stride` times, starting at
    /// range gate `mapping.start`.
    Mapped {
        /// The model field the values come from.
        source: FieldSource,
        /// The field's buffer, `[nrays × native_gates]` in storage row order.
        native: ArrayRef<'a>,
        /// Output rows (the sweep's ray count).
        nrays: usize,
        /// Gates per row of the field.
        native_gates: usize,
        /// Where the native gates sit on the range dimension.
        mapping: GateMapping,
        /// Length of the range dimension.
        out_gates: usize,
        /// The value that pads gates the field does not cover.
        fill: Scalar,
        /// Storage row of each output row.
        rows: RowOrder,
    },
    /// Small computed or reordered arrays (range centres, ray coordinates in
    /// sorted order, volume constants broadcast to rays).
    Owned(ArrayBuf),
    /// A scalar variable.
    Scalar(Scalar),
    /// A text variable.
    Text(Cow<'a, str>),
}

/// Ray order of a sweep's variables relative to storage order.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RowOrder {
    /// Output rows are storage rows.
    Identity,
    /// `permutation[i]` is the storage row shown at position `i`. One `Arc` is
    /// shared by every variable of the sweep.
    Permutation(Arc<[u32]>),
}

impl RowOrder {
    /// Storage row shown at position `row`.
    pub fn source_row(&self, row: usize) -> Option<usize> {
        match self {
            Self::Identity => Some(row),
            Self::Permutation(order) => order.get(row).map(|r| *r as usize),
        }
    }
}

/// A model field: `volume.sweeps[sweep].fields[field]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FieldSource {
    /// Index of the sweep in `Volume::sweeps`.
    pub sweep: u32,
    /// Index of the field in `Sweep::fields`.
    pub field: u32,
}

/// A borrowed contiguous array. Exhaustive, like the
/// [`ArrayBuf`] it borrows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArrayRef<'a> {
    /// Unsigned 8-bit values.
    U8(&'a [u8]),
    /// Unsigned 16-bit values.
    U16(&'a [u16]),
    /// Signed 8-bit values.
    I8(&'a [i8]),
    /// Signed 16-bit values.
    I16(&'a [i16]),
    /// Signed 32-bit values.
    I32(&'a [i32]),
    /// 32-bit floats.
    F32(&'a [f32]),
    /// 64-bit floats.
    F64(&'a [f64]),
}

impl ArrayRef<'_> {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// NumPy-style dtype name.
    pub fn dtype(&self) -> &'static str {
        match self {
            Self::U8(_) => "uint8",
            Self::U16(_) => "uint16",
            Self::I8(_) => "int8",
            Self::I16(_) => "int16",
            Self::I32(_) => "int32",
            Self::F32(_) => "float32",
            Self::F64(_) => "float64",
        }
    }

    /// Copy into an owned buffer.
    pub fn to_array(&self) -> ArrayBuf {
        match self {
            Self::U8(v) => ArrayBuf::U8(v.to_vec()),
            Self::U16(v) => ArrayBuf::U16(v.to_vec()),
            Self::I8(v) => ArrayBuf::I8(v.to_vec()),
            Self::I16(v) => ArrayBuf::I16(v.to_vec()),
            Self::I32(v) => ArrayBuf::I32(v.to_vec()),
            Self::F32(v) => ArrayBuf::F32(v.to_vec()),
            Self::F64(v) => ArrayBuf::F64(v.to_vec()),
        }
    }
}

/// Reorder every sweep's rays in storage into the order a view with
/// `options` shows them ([`crate::model::Sweep::permute_rays`]), so that the
/// view afterwards has [`RowOrder::Identity`] and every field whose native
/// gates are the sweep's range gates is zero-copy
/// ([`DataRef::is_zero_copy`]).
///
/// This is for a binding that moves field buffers into NumPy under a ray
/// order other than the storage order, xradar's default `first_dim="auto"`
/// above all (design note 12.2): under it no Level II, CfRadial or JMA field
/// is zero-copy, because those sweeps start at an arbitrary azimuth. After
/// this call the moved buffer is the variable. The view built afterwards
/// shows the same values and attributes as the one before, per-ray
/// attribute arrays included (both show them in the view's ray order,
/// [`crate::model::AttrValue::ray_alignment`]); rows move in place (a sweep
/// that starts mid-circle is one block rotation per field), so memory does
/// not double. Storage order no longer is the source order afterwards, so a
/// caller that keeps using the volume in Rust should not call this.
///
/// A sweep that holds a verbatim array attribute with one entry per ray
/// under a name not known to be per ray
/// ([`crate::model::RayAlignment::Unknown`],
/// [`crate::model::Sweep::unknown_ray_attribute`]) stays in storage order:
/// reordering it could scramble or misalign that array. The view is the
/// same either way; that sweep's fields are just not zero-copy. Such sweeps
/// are returned, so a binding can report the copies they still need.
///
/// # Errors
///
/// A [`SweepError`] when a sweep's per-ray items do not have one entry per
/// ray. Every sweep is checked before any is reordered, so the volume is
/// unchanged then.
pub fn order_rays_for_view(
    volume: &mut Volume,
    options: ViewOptions,
) -> Result<Vec<UnorderedSweep>, SweepError> {
    let source_format = volume.provenance.source_format;
    let mut unordered = Vec::new();
    // Check every sweep first (4 bytes per ray of the volume for the
    // orders), then move rows: a failure leaves no sweep reordered.
    let mut planned = Vec::new();
    for (index, sweep) in volume.sweeps.iter().enumerate() {
        if let (_, RowOrder::Permutation(order)) =
            build::view_ray_order(sweep, source_format, options)
        {
            if let Some(attribute) = sweep.unknown_ray_attribute() {
                unordered.push(UnorderedSweep {
                    sweep: index,
                    attribute,
                });
                continue;
            }
            sweep.check_permutation(&order)?;
            planned.push((index, order));
        }
    }
    for (index, order) in planned {
        if let Some(sweep) = volume.sweeps.get_mut(index) {
            sweep.permute_rays(&order)?;
        }
    }
    Ok(unordered)
}

/// A sweep that [`order_rays_for_view`] left in storage order although the
/// view shows its rays in another order, so its fields are not zero-copy
/// in that view.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnorderedSweep {
    /// Index of the sweep in [`Volume::sweeps`].
    pub sweep: usize,
    /// The verbatim attribute that kept it in storage order, as `name` or
    /// `field/name` ([`crate::model::Sweep::unknown_ray_attribute`]).
    pub attribute: String,
}

/// Non-fatal conditions a CF writer reports.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ViewWarning {
    /// Ray times are not strictly increasing even in acquisition order (the
    /// source has no per-ray times). The `time` coordinate is written anyway,
    /// as xradar writes it.
    NonMonotonicTime {
        /// Index of the sweep.
        sweep: u32,
    },
}

/// Why a view could not be built.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ViewError {
    /// An attribute's value does not fit the variable's packed type.
    #[error("{path}: attribute {attr} value does not fit the variable's type")]
    OutOfRange {
        /// Path of the variable (`sweep_0/DBZH`).
        path: String,
        /// Name of the attribute.
        attr: &'static str,
    },
}

/// Format-specific attributes a decoder crate contributes (for example xradar's
/// NEXRAD root attributes from `NexradMetadata`).
pub trait ExtraAttrs {
    /// Attributes to add to the root group.
    fn root_attrs(&self, flavor: Flavor) -> Vec<(Cow<'static, str>, AttrValue)>;
    /// Attributes to add to the group of sweep `sweep`.
    fn sweep_attrs(&self, sweep: usize, flavor: Flavor) -> Vec<(Cow<'static, str>, AttrValue)>;
}

impl<'a> VolumeView<'a> {
    /// The volume the view borrows.
    pub fn volume(&self) -> &'a Volume {
        self.volume
    }

    /// A group by slash-separated path (`""` or `"/"` for the root,
    /// `"sweep_0"`, `"sweep_0/monitoring"`).
    pub fn group(&self, path: &str) -> Option<&Group<'a>> {
        let mut group = &self.root;
        for part in path.split('/').filter(|part| !part.is_empty()) {
            group = group.child(part)?;
        }
        Some(group)
    }

    /// The same tree with no borrows (section 12.2).
    pub fn layout(&self) -> VolumeLayout {
        layout::layout(self)
    }
}

impl<'a> Group<'a> {
    /// The child group named `name`.
    pub fn child(&self, name: &str) -> Option<&Group<'a>> {
        self.children.iter().find(|child| child.name == name)
    }

    /// The variable named `name`.
    pub fn variable(&self, name: &str) -> Option<&Variable<'a>> {
        self.variables.iter().find(|variable| variable.name == name)
    }

    /// The attribute named `name`.
    pub fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attrs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    }

    /// Length of the dimension named `name`.
    pub fn dim(&self, name: &str) -> Option<usize> {
        self.dims
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, len)| *len)
    }
}

impl Variable<'_> {
    /// The attribute named `name`.
    pub fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attrs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    }
}

impl Values<'_> {
    /// The encoded values as one owned array in the variable's shape (row-major).
    /// `None` for scalars and text. Mapped fields are padded, repeated and
    /// reordered here; this is the only copy the view makes.
    pub fn materialize(&self) -> Option<ArrayBuf> {
        match self {
            Self::Borrowed(array) => Some(array.to_array()),
            Self::Owned(array) => Some(array.clone()),
            Self::Mapped {
                native,
                nrays,
                native_gates,
                mapping,
                out_gates,
                fill,
                rows,
                ..
            } => Some(layout::apply_mapping(
                *native,
                *nrays,
                *native_gates,
                *mapping,
                *out_gates,
                *fill,
                rows,
            )),
            Self::Scalar(_) | Self::Text(_) => None,
        }
    }
}
