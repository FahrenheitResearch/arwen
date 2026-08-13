//! Shape-aware array operations the paired-run metrics are built from.
//!
//! Every operation here works on the trailing two axes of a field and leaves
//! the leading ones alone, which is what lets one implementation serve a
//! two-dimensional mass field and a staggered three-dimensional wind
//! component without either being special-cased.
//!
//! Two details exist for parity with the reference implementation rather than
//! because they are the obvious choice, and both are called out where they
//! are made:
//!
//! * [`Field::boxcar`] runs the same *prefix-sum* the reference runs -- an
//!   edge-extended pad, a sequential cumulative sum, a difference of two
//!   offsets, a division -- rather than a direct windowed mean.  A cumulative
//!   sum over a long row loses low-order bits in a way a windowed mean does
//!   not, so the two do not agree to the last bit and the reference's shape is
//!   what the numbers were measured under;
//! * the slicing operations produce a *packed* result in row-major order,
//!   because the reference's sums run over the packed temporary its slicing
//!   produces, and a pairwise summation tree is laid over that packing.

use rayon::prelude::*;

/// A field and the shape it carries, stored row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    shape: Vec<usize>,
    values: Vec<f64>,
}

/// Why a field operation could not be carried out.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ShapeError {
    #[error("a field needs at least two axes, got {shape:?}")]
    TooFewAxes { shape: Vec<usize> },

    #[error("shape {shape:?} describes {expected} values but {actual} were supplied")]
    ValueCount {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    #[error("a {rows}x{columns} field is too small for a {width}-cell margin")]
    MarginTooWide {
        rows: usize,
        columns: usize,
        width: usize,
    },

    #[error("a boxcar width must be a positive odd cell count, got {width}")]
    EvenWidth { width: usize },
}

impl Field {
    /// Wrap `values` as a field of `shape`, checking that they agree.
    pub fn new(shape: Vec<usize>, values: Vec<f64>) -> Result<Self, ShapeError> {
        if shape.len() < 2 {
            return Err(ShapeError::TooFewAxes { shape });
        }
        let expected: usize = shape.iter().product();
        if expected != values.len() {
            return Err(ShapeError::ValueCount {
                expected,
                actual: values.len(),
                shape,
            });
        }
        Ok(Self { shape, values })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f64> {
        self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The field as a stack of `rows`x`columns` planes.
    fn planes(&self) -> (usize, usize, usize) {
        let rank = self.shape.len();
        let rows = self.shape[rank - 2];
        let columns = self.shape[rank - 1];
        let stack: usize = self.shape[..rank - 2].iter().product();
        (stack, rows, columns)
    }

    /// Odd-width, edge-extended, separable boxcar mean over the trailing two
    /// axes: the last axis first, then the one before it.
    ///
    /// A width of one is the identity, and the reference returns the array
    /// untouched rather than running a one-wide window over it, so a domain
    /// whose grid spacing already exceeds the filter width costs nothing.
    pub fn boxcar(&self, width: usize) -> Result<Field, ShapeError> {
        if width == 0 || width % 2 == 0 {
            return Err(ShapeError::EvenWidth { width });
        }
        if width == 1 {
            return Ok(self.clone());
        }
        let (_, rows, columns) = self.planes();
        let mut values = self.values.clone();

        // Last axis: every row of every plane is independent.
        values.par_chunks_mut(columns).for_each(|row| {
            let mut scratch = vec![0.0f64; row.len()];
            scratch.copy_from_slice(row);
            smooth_line(&scratch, row, 1, width);
        });

        // Second-to-last axis: every plane is independent, and inside a plane
        // every column is.  Planes are the coarser unit and the one that
        // keeps each worker on one contiguous region of memory.
        values.par_chunks_mut(rows * columns).for_each(|plane| {
            let scratch = plane.to_vec();
            for column in 0..columns {
                smooth_line(
                    &scratch[column..],
                    &mut plane[column..],
                    columns,
                    width,
                );
            }
        });

        Ok(Field {
            shape: self.shape.clone(),
            values,
        })
    }

    /// Drop `width` cells from each side of the trailing two axes, packed.
    pub fn interior(&self, width: usize) -> Result<Field, ShapeError> {
        let (stack, rows, columns) = self.planes();
        if rows <= 2 * width || columns <= 2 * width {
            return Err(ShapeError::MarginTooWide {
                rows,
                columns,
                width,
            });
        }
        let (kept_rows, kept_columns) = (rows - 2 * width, columns - 2 * width);
        let mut values = Vec::with_capacity(stack * kept_rows * kept_columns);
        for plane in 0..stack {
            let base = plane * rows * columns;
            for row in width..rows - width {
                let start = base + row * columns + width;
                values.extend_from_slice(&self.values[start..start + kept_columns]);
            }
        }
        let rank = self.shape.len();
        let mut shape = self.shape.clone();
        shape[rank - 2] = kept_rows;
        shape[rank - 1] = kept_columns;
        Field::new(shape, values)
    }

    /// The outer `width`-cell frame of the trailing two axes: one row per
    /// selected cell, one column per plane.
    ///
    /// The cell-first arrangement is not a preference, it is the reference's
    /// memory layout.  Selecting the trailing axes of a stack of planes with
    /// a boolean mask leaves the reference holding a *column-major* array --
    /// logically plane-by-cell, physically cell-by-plane -- and its sums walk
    /// memory rather than logic.  So a ring packed plane-first would be summed
    /// in a different order than the reference sums it and would land a bit
    /// away on a large enough field, which is exactly what it did before this
    /// was traced.
    ///
    /// Nothing downstream cares which way round it is: the boundary metric
    /// combines two arms and two times element-wise at matching positions,
    /// and every one of them is packed by this same function.
    pub fn boundary_values(&self, width: usize) -> Result<Field, ShapeError> {
        let (stack, rows, columns) = self.planes();
        if rows <= 2 * width || columns <= 2 * width {
            return Err(ShapeError::MarginTooWide {
                rows,
                columns,
                width,
            });
        }
        let selected = boundary_offsets(rows, columns, width);
        let mut values = Vec::with_capacity(stack * selected.len());
        for &offset in &selected {
            for plane in 0..stack {
                values.push(self.values[plane * rows * columns + offset]);
            }
        }
        Field::new(vec![selected.len(), stack], values)
    }

    /// Reduce a stack of planes to its per-column maximum, propagating a bad
    /// cell up its own column.  A field that is already one plane is its own
    /// composite.
    pub fn composite(&self) -> Field {
        let (stack, rows, columns) = self.planes();
        if stack == 1 {
            return Field {
                shape: vec![rows, columns],
                values: self.values.clone(),
            };
        }
        Field {
            shape: vec![rows, columns],
            values: crate::stats::column_max(&self.values, stack, rows * columns),
        }
    }
}

/// Offsets into one plane of the cells within `width` of any edge, in
/// row-major order.
fn boundary_offsets(rows: usize, columns: usize, width: usize) -> Vec<usize> {
    let mut offsets = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            let nearest = row
                .min(rows - 1 - row)
                .min(column)
                .min(columns - 1 - column);
            if nearest < width {
                offsets.push(row * columns + column);
            }
        }
    }
    offsets
}

/// Smooth one strided line in place through the reference's prefix sum.
///
/// `source` and `target` are the same line before and after; `stride` is the
/// gap between neighbours along the axis being smoothed.  The line is padded
/// by repeating its end values, accumulated once from the start, and read
/// back as a difference of two prefix offsets divided by the width -- the
/// reference's arithmetic in the reference's order.
fn smooth_line(source: &[f64], target: &mut [f64], stride: usize, width: usize) {
    let count = source.len().div_ceil(stride);
    let radius = width / 2;
    let padded = count + 2 * radius;
    let mut prefix = Vec::with_capacity(padded + 1);
    prefix.push(0.0f64);
    let mut running = 0.0f64;
    for step in 0..padded {
        let index = (step as isize - radius as isize).clamp(0, count as isize - 1) as usize;
        running += source[index * stride];
        prefix.push(running);
    }
    let span = width as f64;
    for step in 0..count {
        target[step * stride] = (prefix[step + width] - prefix[step]) / span;
    }
}

/// Nearest odd cell count spanning `physical_width_m` at `dx_m`.
pub fn odd_width_cells(physical_width_m: f64, dx_m: f64) -> usize {
    let cells = (physical_width_m / dx_m + 0.5).floor().max(1.0) as usize;
    if cells % 2 == 1 {
        cells
    } else {
        cells + 1
    }
}

/// Neighbourhood half-width in cells at one domain's grid spacing.
pub fn half_width_cells(physical_radius_m: f64, dx_m: f64) -> usize {
    (physical_radius_m / dx_m + 0.5).floor().max(0.0) as usize
}

/// Cell count covering `min_area_km2` at one domain's grid spacing.
pub fn minimum_object_cells(min_area_km2: f64, dx_m: f64) -> usize {
    ((min_area_km2 * 1.0e6 / (dx_m * dx_m)).ceil() as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_one_wide_boxcar_is_the_identity() {
        let field = Field::new(vec![3, 4], (0..12).map(|i| i as f64).collect()).expect("shape");
        assert_eq!(field.boxcar(1).expect("width"), field);
    }

    #[test]
    fn a_boxcar_of_a_constant_field_is_that_constant() {
        // Edge extension is what makes this true at the corners as well as in
        // the middle; a zero-padded filter would dip at every edge.
        let field = Field::new(vec![2, 5, 7], vec![4.5; 70]).expect("shape");
        for value in field.boxcar(3).expect("width").values() {
            assert!((value - 4.5).abs() < 1.0e-15);
        }
    }

    /// Oracle: the reference's separable boxcar on a 1..=12 ramp shaped 3x4
    /// at width 3, computed by hand from the same pad/cumsum/difference rule.
    #[test]
    fn a_boxcar_matches_a_hand_worked_window() {
        let field = Field::new(vec![3, 4], (1..=12).map(|i| i as f64).collect()).expect("shape");
        let smoothed = field.boxcar(3).expect("width");
        // Row means after the last-axis pass are [1.333, 2, 3, 3.667] on the
        // first row, and the second-axis pass then averages rows [0,0,1] on
        // the top row, so the top-left cell is (1.333*2 + 5.333)/3.
        let expected_first = ((1.0 + 1.0 + 2.0) / 3.0 * 2.0 + (5.0 + 5.0 + 6.0) / 3.0) / 3.0;
        assert!((smoothed.values()[0] - expected_first).abs() < 1.0e-14);
    }

    #[test]
    fn the_interior_drops_a_margin_from_every_plane() {
        let field = Field::new(vec![2, 4, 5], (0..40).map(|i| i as f64).collect()).expect("shape");
        let inner = field.interior(1).expect("margin");
        assert_eq!(inner.shape(), [2, 2, 3]);
        assert_eq!(inner.values(), [6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 26.0, 27.0, 28.0, 31.0, 32.0, 33.0]);
    }

    #[test]
    fn the_boundary_is_everything_the_interior_is_not() {
        let field = Field::new(vec![1, 4, 5], (0..20).map(|i| i as f64).collect()).expect("shape");
        let ring = field.boundary_values(1).expect("margin");
        let inner = field.interior(1).expect("margin");
        assert_eq!(ring.len() + inner.len(), field.len());
        // Row-major order of the selected cells, not an edge-walk order.
        assert_eq!(&ring.values()[..6], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// The ring is packed cell-first, so the cells of one column of the
    /// stack sit together.  This is the reference's memory layout and
    /// therefore the order its sums run in; packing plane-first instead
    /// moves the last bit of a large boundary metric.
    #[test]
    fn the_boundary_packs_a_stack_cell_first() {
        // Two planes of a 3x3 field, offset by a hundred so the plane a
        // value came from is readable in the value.
        let values: Vec<f64> = (0..18).map(|i| ((i / 9) * 100 + (i % 9)) as f64).collect();
        let ring = Field::new(vec![2, 3, 3], values)
            .expect("shape")
            .boundary_values(1)
            .expect("margin");
        // Every cell of a 3x3 field but the middle one is within one of an
        // edge, so the ring is eight cells read across the two planes.
        assert_eq!(ring.shape(), [8, 2]);
        assert_eq!(
            ring.values(),
            [0.0, 100.0, 1.0, 101.0, 2.0, 102.0, 3.0, 103.0,
             5.0, 105.0, 6.0, 106.0, 7.0, 107.0, 8.0, 108.0]
        );
    }

    #[test]
    fn a_margin_wider_than_the_field_is_refused_rather_than_clamped() {
        let field = Field::new(vec![4, 4], vec![0.0; 16]).expect("shape");
        assert!(field.interior(2).is_err());
        assert!(field.boundary_values(2).is_err());
        assert!(field.interior(1).is_ok());
    }

    #[test]
    fn the_composite_takes_each_columns_largest_level() {
        let field = Field::new(vec![2, 2, 2], vec![1.0, 9.0, 3.0, 4.0, 5.0, 2.0, 7.0, 0.0])
            .expect("shape");
        assert_eq!(field.composite().values(), [5.0, 9.0, 7.0, 4.0]);
        assert_eq!(field.composite().shape(), [2, 2]);
    }

    /// The cell-count rules decide filter widths and object thresholds, so a
    /// half-cell drift in any of them silently rescores a whole domain.
    #[test]
    fn cell_counts_round_the_way_the_reference_rounds() {
        // Six kilometres of filter across the registered spacing ladder.
        assert_eq!(odd_width_cells(6000.0, 12000.0), 1);
        assert_eq!(odd_width_cells(6000.0, 3000.0), 3);
        assert_eq!(odd_width_cells(6000.0, 1000.0), 7);
        assert_eq!(odd_width_cells(6000.0, 1000.0 / 3.0), 19);
        // Five kilometres of neighbourhood radius at the same spacings.
        assert_eq!(half_width_cells(5000.0, 12000.0), 0);
        assert_eq!(half_width_cells(5000.0, 1000.0 / 3.0), 15);
        // Twenty-five square kilometres of object.  At a third of a
        // kilometre the exact quotient is 225, but the spacing is not
        // representable, so the quotient lands a hair above it and the
        // ceiling takes 226 rather than 225.  The reference lands there too.
        assert_eq!(minimum_object_cells(25.0, 12000.0), 1);
        assert_eq!(minimum_object_cells(25.0, 1000.0), 25);
        assert_eq!(minimum_object_cells(25.0, 1000.0 / 3.0), 226);
    }
}
