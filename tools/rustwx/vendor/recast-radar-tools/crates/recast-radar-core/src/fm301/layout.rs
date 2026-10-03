//! The borrow-free form of a [`VolumeView`] (design note 12.2) and the gate
//! mapping applied on read.

use super::{ArrayRef, FieldSource, Group, RowOrder, Values, ViewWarning, VolumeView};
use crate::model::{ArrayBuf, AttrValue, GateMapping, Scalar};

/// A [`VolumeView`] with no borrows: every dataset variable is a
/// [`DataRef::Field`], other values are copied (they are small).
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeLayout {
    /// The root group, with every group below it.
    pub root: GroupLayout,
    /// Non-fatal conditions found while building the view.
    pub warnings: Vec<ViewWarning>,
}

/// One group of a [`VolumeLayout`], owning its names and small values.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupLayout {
    /// Group name (`""` for the root, `sweep_0`, `monitoring`, ...).
    pub name: String,
    /// Dimension names and lengths.
    pub dims: Vec<(String, usize)>,
    /// The group's variables.
    pub variables: Vec<VariableLayout>,
    /// The group's attributes, typed.
    pub attrs: Vec<(String, AttrValue)>,
    /// Child groups.
    pub children: Vec<GroupLayout>,
}

/// One variable of a [`GroupLayout`].
#[derive(Clone, Debug, PartialEq)]
pub struct VariableLayout {
    /// Variable name.
    pub name: String,
    /// Dimension names, outermost first.
    pub dims: Vec<String>,
    /// Where the values come from.
    pub data: DataRef,
    /// The variable's attributes, typed.
    pub attrs: Vec<(String, AttrValue)>,
}

/// Where a variable's values come from.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DataRef {
    /// A model field's buffer, `[nrays × native_gates]` in storage row order,
    /// shown through `rows` and `mapping` on an `out_gates` range.
    Field {
        /// The model field the values come from.
        source: FieldSource,
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
    /// Values computed or copied by the view (coordinates, small variables).
    Array(ArrayBuf),
    /// A scalar variable.
    Scalar(Scalar),
    /// A text variable.
    Text(String),
}

impl DataRef {
    /// `true` when the field's buffer, reshaped to `[nrays, native_gates]`, is
    /// already the variable (no reordering, padding or repetition).
    pub fn is_zero_copy(&self) -> bool {
        matches!(
            self,
            Self::Field { native_gates, mapping, out_gates, rows: RowOrder::Identity, .. }
                if *mapping == GateMapping::IDENTITY && native_gates == out_gates
        )
    }
}

impl GroupLayout {
    /// The child group named `name`.
    pub fn child(&self, name: &str) -> Option<&GroupLayout> {
        self.children.iter().find(|child| child.name == name)
    }

    /// The variable named `name`.
    pub fn variable(&self, name: &str) -> Option<&VariableLayout> {
        self.variables.iter().find(|variable| variable.name == name)
    }
}

pub(super) fn layout(view: &VolumeView<'_>) -> VolumeLayout {
    VolumeLayout {
        root: group_layout(view, &view.root),
        warnings: view.warnings.clone(),
    }
}

fn group_layout(view: &VolumeView<'_>, group: &Group<'_>) -> GroupLayout {
    GroupLayout {
        name: group.name.to_string(),
        dims: group
            .dims
            .iter()
            .map(|(name, len)| (name.to_string(), *len))
            .collect(),
        variables: group
            .variables
            .iter()
            .map(|variable| VariableLayout {
                name: variable.name.to_string(),
                dims: variable.dims.iter().map(|dim| dim.to_string()).collect(),
                data: data_ref(view, variable.source, &variable.values),
                attrs: variable
                    .attrs
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.clone()))
                    .collect(),
            })
            .collect(),
        attrs: group
            .attrs
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect(),
        children: group
            .children
            .iter()
            .map(|child| group_layout(view, child))
            .collect(),
    }
}

fn data_ref(view: &VolumeView<'_>, source: Option<FieldSource>, values: &Values<'_>) -> DataRef {
    match values {
        Values::Mapped {
            source,
            nrays,
            native_gates,
            mapping,
            out_gates,
            fill,
            rows,
            ..
        } => DataRef::Field {
            source: *source,
            nrays: *nrays,
            native_gates: *native_gates,
            mapping: *mapping,
            out_gates: *out_gates,
            fill: *fill,
            rows: rows.clone(),
        },
        Values::Borrowed(array) => {
            let field = source.and_then(|source| {
                view.volume
                    .sweeps
                    .get(source.sweep as usize)?
                    .fields
                    .get(source.field as usize)
                    .map(|field| (source, field))
            });
            match field {
                Some((source, field)) => DataRef::Field {
                    source,
                    nrays: field.nrays as usize,
                    native_gates: field.ngates as usize,
                    mapping: GateMapping::IDENTITY,
                    out_gates: field.ngates as usize,
                    fill: field.data.fill_scalar(),
                    rows: RowOrder::Identity,
                },
                None => DataRef::Array(array.to_array()),
            }
        }
        Values::Owned(array) => DataRef::Array(array.clone()),
        Values::Scalar(scalar) => DataRef::Scalar(*scalar),
        Values::Text(text) => DataRef::Text(text.to_string()),
    }
}

/// Apply a field mapping: `nrays` output rows in `rows` order, native gates
/// repeated `mapping.stride` times from range gate `mapping.start`, padded with
/// `fill` to `out_gates`.
pub(super) fn apply_mapping(
    native: ArrayRef<'_>,
    nrays: usize,
    native_gates: usize,
    mapping: GateMapping,
    out_gates: usize,
    fill: Scalar,
    rows: &RowOrder,
) -> ArrayBuf {
    let map = MapSpec {
        nrays,
        native_gates,
        mapping,
        out_gates,
        rows,
    };
    match native {
        ArrayRef::U8(values) => ArrayBuf::U8(map.apply(values, scalar_or(fill, 0))),
        ArrayRef::U16(values) => ArrayBuf::U16(map.apply(values, scalar_or(fill, 0))),
        ArrayRef::I8(values) => ArrayBuf::I8(map.apply(values, scalar_or(fill, 0))),
        ArrayRef::I16(values) => ArrayBuf::I16(map.apply(values, scalar_or(fill, 0))),
        ArrayRef::I32(values) => ArrayBuf::I32(map.apply(values, scalar_or(fill, 0))),
        ArrayRef::F32(values) => ArrayBuf::F32(map.apply(values, scalar_or(fill, f32::NAN))),
        ArrayRef::F64(values) => ArrayBuf::F64(map.apply(values, scalar_or(fill, f64::NAN))),
    }
}

trait FromScalar: Sized {
    fn from_scalar(scalar: Scalar) -> Option<Self>;
}

macro_rules! from_scalar {
    ($t:ty, $variant:ident) => {
        impl FromScalar for $t {
            fn from_scalar(scalar: Scalar) -> Option<Self> {
                match scalar {
                    Scalar::$variant(value) => Some(value),
                    _ => None,
                }
            }
        }
    };
}

from_scalar!(u8, U8);
from_scalar!(u16, U16);
from_scalar!(i8, I8);
from_scalar!(i16, I16);
from_scalar!(i32, I32);
from_scalar!(f32, F32);
from_scalar!(f64, F64);

fn scalar_or<T: FromScalar>(scalar: Scalar, default: T) -> T {
    T::from_scalar(scalar).unwrap_or(default)
}

struct MapSpec<'r> {
    nrays: usize,
    native_gates: usize,
    mapping: GateMapping,
    out_gates: usize,
    rows: &'r RowOrder,
}

impl MapSpec<'_> {
    /// Each output value is written once: per row, `fill` up to
    /// `mapping.start`, the native gates (a block copy at stride 1, each gate
    /// repeated `stride` times otherwise), then `fill` to `out_gates`. A row
    /// whose source row is missing is all `fill`.
    fn apply<T: Copy>(&self, native: &[T], fill: T) -> Vec<T> {
        let out_gates = self.out_gates;
        let mut out = Vec::with_capacity(self.nrays.saturating_mul(out_gates));
        if out_gates == 0 {
            return out;
        }
        let stride = self.mapping.stride.max(1) as usize;
        let start = self.mapping.start as usize;
        for out_row in 0..self.nrays {
            let row = self.rows.source_row(out_row).and_then(|source_row| {
                let begin = source_row.saturating_mul(self.native_gates);
                native.get(begin..begin.saturating_add(self.native_gates))
            });
            let row_start = out.len();
            if let Some(row) = row
                && start < out_gates
            {
                out.resize(row_start + start, fill);
                // Native gates whose first output gate is inside the row.
                let shown = (out_gates - start).div_ceil(stride).min(row.len());
                if stride == 1 {
                    out.extend_from_slice(&row[..shown]);
                } else {
                    for (gate, value) in row[..shown].iter().enumerate() {
                        let first = start + gate * stride;
                        let count = stride.min(out_gates - first);
                        out.extend(std::iter::repeat_n(*value, count));
                    }
                }
            }
            out.resize(row_start + out_gates, fill);
        }
        out
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn mapping_pads_repeats_and_reorders() {
        let native = [1u8, 2, 3, 4];
        let out = apply_mapping(
            ArrayRef::U8(&native),
            2,
            2,
            GateMapping {
                start: 1,
                stride: 2,
            },
            6,
            Scalar::U8(0),
            &RowOrder::Permutation(Arc::from(vec![1u32, 0])),
        );
        assert_eq!(out, ArrayBuf::U8(vec![0, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2, 0]));
    }

    /// The row-wise writer against a direct per-gate construction, over
    /// starts, strides, native and output widths that clip, pad and
    /// repeat, and row orders with missing source rows.
    #[test]
    fn mapping_matches_a_per_gate_reference() {
        let native: Vec<u16> = (1..=3 * 5).collect();
        let orders = [
            RowOrder::Identity,
            RowOrder::Permutation(Arc::from(vec![2u32, 0, 1])),
            RowOrder::Permutation(Arc::from(vec![1u32, 7, 0])),
        ];
        for rows in &orders {
            for start in 0..8u32 {
                for stride in 0..4u32 {
                    for out_gates in 0..14usize {
                        let mapping = GateMapping { start, stride };
                        let got = apply_mapping(
                            ArrayRef::U16(&native),
                            3,
                            5,
                            mapping,
                            out_gates,
                            Scalar::U16(9),
                            rows,
                        );
                        let mut want = vec![9u16; 3 * out_gates];
                        let step = stride.max(1) as usize;
                        for out_row in 0..3 {
                            let Some(source) = rows.source_row(out_row).filter(|r| *r < 3) else {
                                continue;
                            };
                            for gate in 0..5 {
                                for rep in 0..step {
                                    let position = start as usize + gate * step + rep;
                                    if position < out_gates {
                                        want[out_row * out_gates + position] =
                                            native[source * 5 + gate];
                                    }
                                }
                            }
                        }
                        assert_eq!(
                            got,
                            ArrayBuf::U16(want),
                            "start {start} stride {stride} out {out_gates} rows {rows:?}"
                        );
                    }
                }
            }
        }
    }
}
