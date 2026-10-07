//! Radar no-coverage on comparison sheets.
//!
//! An observed radar panel has two kinds of empty cell, and they mean
//! opposite things.  MRMS writes `-99` where a radar saw the column and
//! found no echo (an observation: "no rain here") and `-999` where no radar
//! saw the column at all (outside the network, or a radar that is down).
//! Drawn the same way, both are blank, and an outage reads as "no rain":
//! a DA member's echo over a dead radar's footprint looks like a false
//! alarm when nothing was there to observe it.
//!
//! So a sheet that carries an observed radar panel draws the unobserved
//! cells in flat light grey on that panel, and the same footprint as a thin
//! grey outline on every model panel, so a false-echo judgement is visibly
//! limited to covered ground.  Which source values are "unobserved" is
//! table metadata (`unobserved_values` in `comparison-observations.json`,
//! or the pack's own validity mask), never a code path per source.
//!
//! The coverage mask lives on the sheet's run grid, row-major `ny * nx`,
//! `true` where nothing was observed.  A run-grid point that falls outside
//! the observation lattice is unobserved too.

use rustwx_render::{Color, ContourLayer, ContourLinePattern};

/// The fill an unobserved cell wears on the observed panel.
pub const NO_COVERAGE_FILL: Color = Color::rgba(208, 208, 208, 255);
/// The outline of the unobserved footprint on every other panel.
pub const NO_COVERAGE_LINE: Color = Color::rgba(110, 110, 110, 255);
/// The sheet header's legend entry.
pub const NO_COVERAGE_LEGEND: &str = "grey = no radar coverage";

/// How one panel shows the sheet's no-coverage footprint.
#[derive(Clone, Copy, Debug, Default)]
pub enum CoverageDraw<'a> {
    /// The sheet carries no observed panel: nothing is drawn.
    #[default]
    None,
    /// The observed panel: unobserved cells filled light grey under the data.
    Fill(&'a [bool]),
    /// A model panel: the unobserved footprint as a thin grey outline.
    Outline(&'a [bool]),
}

/// Fold the coverage mask into one plane so it rides through a value-only
/// nearest-point mapping (`verification::mapped_fields`) in one pass.
///
/// Unobserved cells become NaN, which is also what the mapping writes for a
/// target outside the source lattice, so both come out unobserved.  An
/// observed cell with no value to draw (MRMS `-99`, no echo) becomes
/// negative infinity, which the mapping copies like any other number.
pub fn encode_for_mapping(values: &[f32], unobserved: &[bool]) -> Vec<f64> {
    values
        .iter()
        .zip(unobserved)
        .map(|(value, unobserved)| {
            if *unobserved {
                f64::NAN
            } else if value.is_nan() {
                f64::NEG_INFINITY
            } else {
                f64::from(*value)
            }
        })
        .collect()
}

/// Split a mapped [`encode_for_mapping`] plane back into the drawable
/// values (NaN wherever nothing is drawn) and the coverage mask.
pub fn decode_mapped(mapped: &[f64]) -> (Vec<f32>, Vec<bool>) {
    mapped
        .iter()
        .map(|value| {
            if value.is_nan() {
                (f32::NAN, true)
            } else if *value == f64::NEG_INFINITY {
                (f32::NAN, false)
            } else {
                (*value as f32, false)
            }
        })
        .unzip()
}

/// The mask of a plane whose NaN cells are exactly its unobserved cells (a
/// packaged observation, whose no-echo floor is a finite valid value).
pub fn mask_from_nan(values: &[f32]) -> Vec<bool> {
    values.iter().map(|value| value.is_nan()).collect()
}

/// The thin grey outline of the unobserved footprint, or none when there is
/// no edge to draw (every cell observed, or none).
pub fn outline_layer(unobserved: &[bool]) -> Option<ContourLayer> {
    let any = unobserved.iter().any(|cell| *cell);
    let all = unobserved.iter().all(|cell| *cell);
    if !any || all {
        return None;
    }
    Some(ContourLayer {
        data: unobserved.iter().map(|cell| if *cell { 1.0 } else { 0.0 }).collect(),
        levels: vec![0.5],
        color: NO_COVERAGE_LINE,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    })
}

/// A projected coordinate at fractional grid index `(fi, fj)`, bilinear
/// inside the grid and extrapolated half a cell past its edge.
fn at(plane: &[f64], nx: usize, ny: usize, fi: f64, fj: f64) -> f64 {
    let i0 = (fi.floor() as isize).clamp(0, nx.saturating_sub(2) as isize) as usize;
    let j0 = (fj.floor() as isize).clamp(0, ny.saturating_sub(2) as isize) as usize;
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(ny - 1);
    let tx = if i1 == i0 { 0.0 } else { fi - i0 as f64 };
    let ty = if j1 == j0 { 0.0 } else { fj - j0 as f64 };
    let p = |i: usize, j: usize| plane[j * nx + i];
    let bottom = p(i0, j0) + (p(i1, j0) - p(i0, j0)) * tx;
    let top = p(i0, j1) + (p(i1, j1) - p(i0, j1)) * tx;
    bottom + (top - bottom) * ty
}

/// The unobserved cells as projected polygon rings: one ring per run of
/// unobserved cells along a grid row, its edges on the cell boundaries
/// (half a cell either side of the centres) and following the grid between
/// every cell, so a curved row stays on its cells.
///
/// The rings never overlap, so one even-odd fill of all of them paints
/// exactly the unobserved cells.
pub fn fill_rings(
    unobserved: &[bool],
    nx: usize,
    ny: usize,
    projected_x: &[f64],
    projected_y: &[f64],
) -> Vec<Vec<(f64, f64)>> {
    let points = nx * ny;
    if nx == 0
        || ny == 0
        || unobserved.len() != points
        || projected_x.len() != points
        || projected_y.len() != points
    {
        return Vec::new();
    }
    let corner = |fi: f64, fj: f64| {
        (
            at(projected_x, nx, ny, fi, fj),
            at(projected_y, nx, ny, fi, fj),
        )
    };
    let mut rings = Vec::new();
    for j in 0..ny {
        let row = &unobserved[j * nx..(j + 1) * nx];
        let mut i = 0;
        while i < nx {
            if !row[i] {
                i += 1;
                continue;
            }
            let start = i;
            while i < nx && row[i] {
                i += 1;
            }
            let end = i; // exclusive
            let (low, high) = (j as f64 - 0.5, j as f64 + 0.5);
            let mut ring = Vec::with_capacity(2 * (end - start + 1));
            for edge in start..=end {
                ring.push(corner(edge as f64 - 0.5, low));
            }
            for edge in (start..=end).rev() {
                ring.push(corner(edge as f64 - 0.5, high));
            }
            rings.push(ring);
        }
    }
    rings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_echo_and_no_coverage_survive_the_mapping_apart() {
        let values = [f32::NAN, f32::NAN, 32.0, -5.0];
        let unobserved = [true, false, false, false];
        let encoded = encode_for_mapping(&values, &unobserved);
        assert!(encoded[0].is_nan());
        assert_eq!(encoded[1], f64::NEG_INFINITY);
        // A mapping writes NaN where the target is outside the lattice:
        // that point is unobserved too.
        let mapped = [encoded[0], encoded[1], encoded[2], encoded[3], f64::NAN];
        let (drawn, mask) = decode_mapped(&mapped);
        assert_eq!(mask, vec![true, false, false, false, true]);
        assert!(drawn[0].is_nan() && drawn[1].is_nan() && drawn[4].is_nan());
        assert_eq!(drawn[2], 32.0);
        assert_eq!(drawn[3], -5.0);
    }

    #[test]
    fn the_outline_exists_only_where_there_is_an_edge() {
        assert!(outline_layer(&[false; 6]).is_none(), "fully covered: nothing to outline");
        assert!(outline_layer(&[true; 6]).is_none(), "nothing observed: no edge inside the map");
        let layer = outline_layer(&[true, false, false, true]).expect("an edge");
        assert_eq!(layer.data, vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(layer.levels, vec![0.5]);
        assert!(!layer.labels);
        assert_eq!(layer.color, NO_COVERAGE_LINE);
    }

    #[test]
    fn rings_cover_runs_of_unobserved_cells_on_cell_boundaries() {
        // A 4 x 3 unit grid: x = i, y = 10 * j.
        let (nx, ny) = (4, 3);
        let x: Vec<f64> = (0..nx * ny).map(|k| (k % nx) as f64).collect();
        let y: Vec<f64> = (0..nx * ny).map(|k| 10.0 * (k / nx) as f64).collect();
        #[rustfmt::skip]
        let mask = [
            true,  true,  false, true,
            false, false, false, false,
            false, false, false, true,
        ];
        let rings = fill_rings(&mask, nx, ny, &x, &y);
        assert_eq!(rings.len(), 3, "two runs in row 0, one in row 2");
        assert_eq!(
            rings[0],
            vec![(-0.5, -5.0), (0.5, -5.0), (1.5, -5.0), (1.5, 5.0), (0.5, 5.0), (-0.5, 5.0)]
        );
        assert_eq!(rings[1], vec![(2.5, -5.0), (3.5, -5.0), (3.5, 5.0), (2.5, 5.0)]);
        assert_eq!(rings[2], vec![(2.5, 15.0), (3.5, 15.0), (3.5, 25.0), (2.5, 25.0)]);
        assert!(fill_rings(&[false; 12], nx, ny, &x, &y).is_empty());
        assert!(fill_rings(&mask[..11], nx, ny, &x, &y).is_empty(), "a short mask is refused");
    }

    #[test]
    fn a_packaged_plane_is_unobserved_exactly_where_it_is_nan() {
        assert_eq!(mask_from_nan(&[f32::NAN, 0.0, -30.0]), vec![true, false, false]);
    }
}
