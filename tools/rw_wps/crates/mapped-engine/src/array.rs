//! Array steps of the data path: stacking, transposition, unit transform.
//!
//! Every operation keeps float64 arithmetic and the numpy ORDER of
//! operations, because the parity contract compares sha256 of the array
//! bytes.  `values * scale + offset` is a fused-free two-step in numpy and
//! must stay one here: `mul_add` would change the last bit.

use ndarray::{ArrayD, Axis, IxDyn};

use crate::refusal::{frame_invalid, Result};

/// `mapped_source._unit_transform`.
///
/// The finite-check is deliberately taken over the INPUT's finite cells:
/// a cell that arrived missing stays missing, while a cell that was finite
/// and became infinite is a declared-scale defect and refuses.
pub fn unit_transform(
    mut values: ArrayD<f64>,
    scale: f64,
    offset: f64,
    name: &str,
) -> Result<ArrayD<f64>> {
    // In place, in one pass.  The two-array spelling this replaces
    // cloned the whole field to hold the result and then walked both
    // copies again to compare them cell by cell; on a 3-km frameset that
    // is half a gigabyte of copy per field for an answer that depends on
    // one cell at a time.  The arithmetic is unchanged and stays
    // unfused -- `value * scale + offset` in two steps, never `mul_add`,
    // because the parity contract is over the array's bytes.
    let mut overflowed = false;
    for value in values.iter_mut() {
        let source = *value;
        let converted = source * scale + offset;
        overflowed |= source.is_finite() && !converted.is_finite();
        *value = converted;
    }
    if overflowed {
        return Err(frame_invalid(format!(
            "unit conversion for {name} produced non-finite data"
        )));
    }
    Ok(values)
}

/// `mapped_source._transpose_to_target`.
pub fn transpose_to_target(
    values: ArrayD<f64>,
    source_axes: &[String],
    target_axes: &[String],
    name: &str,
) -> Result<ArrayD<f64>> {
    let mut source_sorted: Vec<&String> = source_axes.iter().collect();
    let mut target_sorted: Vec<&String> = target_axes.iter().collect();
    source_sorted.sort();
    target_sorted.sort();
    if source_sorted != target_sorted {
        return Err(frame_invalid(format!(
            "{name} cannot map source axes {source_axes:?} to target axes \
             {target_axes:?} without an explicit stacking dimension"
        )));
    }
    let order: Vec<usize> = target_axes
        .iter()
        .map(|axis| {
            source_axes
                .iter()
                .position(|candidate| candidate == axis)
                .expect("axis sets proven equal above")
        })
        .collect();
    // A source already stated on the target axes is the COMMON case on
    // the GRIB path -- a mapping that declares its own record layout
    // asks for no permutation at all -- and for it the general spelling
    // below permutes by an identity, finds the array already standard,
    // and copies it anyway.  Byte for byte that copy is the array it
    // copied, so it is skipped.
    if order.iter().enumerate().all(|(position, axis)| position == *axis)
        && values.is_standard_layout()
    {
        return Ok(values);
    }
    let permuted = values.permuted_axes(IxDyn(&order));
    Ok(permuted.as_standard_layout().to_owned())
}

/// `numpy.stack(arrays, axis=...)` over equally shaped members.
pub fn stack(members: &[ArrayD<f64>], axis: usize, name: &str) -> Result<ArrayD<f64>> {
    let Some(first) = members.first() else {
        return Err(frame_invalid(format!("{name} has no records to stack")));
    };
    if members
        .iter()
        .any(|member| member.shape() != first.shape())
    {
        return Err(frame_invalid(format!(
            "{name} stacking members do not share one shape"
        )));
    }
    // The unit axis goes in on a VIEW.  Cloning each member to reshape
    // it copied the whole stack once before `concatenate` copied it
    // again; inserting the axis costs a stride, and the concatenation
    // that follows is unchanged, so the stacked array is the same bytes
    // for half the memory traffic.
    let views: Vec<_> = members
        .iter()
        .map(|member| member.view().insert_axis(Axis(axis)))
        .collect();
    ndarray::concatenate(Axis(axis), &views)
        .map_err(|error| frame_invalid(format!("{name} stacking failed: {error}")))
}

/// The elements of an array in C order, which is the order the frame
/// stream and every array digest use.
///
/// BORROWED when the array is already in that order, which on this data
/// path it almost always is: every producer of a canonical field ends in
/// `as_standard_layout().to_owned()`, a stack, or an elementwise pass
/// over a standard array.  The copy this used to make unconditionally
/// was 8 GB per pass on a 3-km CONUS frameset and the frameset is walked
/// three times (frame header digest, stream digest, stream write), so it
/// was the single largest allocation in the engine and bought nothing.
/// A non-standard array still gets its copy — the digest and the stream
/// are defined over C order, and nothing here may hash a transposed
/// view.
pub fn contiguous(values: &ArrayD<f64>) -> std::borrow::Cow<'_, [f64]> {
    match values.as_slice() {
        Some(slice) => std::borrow::Cow::Borrowed(slice),
        None => std::borrow::Cow::Owned(
            values.as_standard_layout().iter().copied().collect::<Vec<f64>>(),
        ),
    }
}

pub fn count_nan(values: &ArrayD<f64>) -> usize {
    values.iter().filter(|value| value.is_nan()).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn transposition_reorders_axes_by_name() {
        let values = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let source = vec!["y".to_owned(), "x".to_owned()];
        let target = vec!["x".to_owned(), "y".to_owned()];
        let moved = transpose_to_target(values, &source, &target, "field").unwrap();
        assert_eq!(moved.shape(), &[3, 2]);
        assert_eq!(contiguous(&moved), vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn a_target_axis_the_source_never_had_refuses() {
        let values = Array::from_shape_vec(IxDyn(&[2]), vec![1., 2.]).unwrap();
        let refusal = transpose_to_target(
            values,
            &["y".to_owned()],
            &["vertical".to_owned()],
            "field",
        )
        .unwrap_err();
        assert!(refusal.message.contains("without an explicit stacking dimension"));
    }

    #[test]
    fn unit_transform_scales_then_offsets_and_leaves_missing_cells_missing() {
        let values = Array::from_shape_vec(IxDyn(&[3]), vec![1.0, f64::NAN, 3.0]).unwrap();
        let converted = unit_transform(values, 100.0, 5.0, "field").unwrap();
        assert_eq!(converted[[0]], 105.0);
        assert!(converted[[1]].is_nan());
        assert_eq!(converted[[2]], 305.0);
    }

    #[test]
    fn a_scale_that_overflows_a_finite_cell_refuses() {
        let values = Array::from_shape_vec(IxDyn(&[1]), vec![1e300]).unwrap();
        let refusal = unit_transform(values, 1e300, 0.0, "field").unwrap_err();
        assert!(refusal.message.contains("non-finite"));
    }

    #[test]
    fn stacking_inserts_the_new_axis_at_the_declared_position() {
        let first = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1., 2., 3., 4.]).unwrap();
        let second = Array::from_shape_vec(IxDyn(&[2, 2]), vec![5., 6., 7., 8.]).unwrap();
        let stacked = stack(&[first, second], 0, "field").unwrap();
        assert_eq!(stacked.shape(), &[2, 2, 2]);
        assert_eq!(contiguous(&stacked), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }
}
