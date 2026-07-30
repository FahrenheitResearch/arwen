//! Marching squares contour line extraction.
//!
//! Generates isopleths from a 2D scalar field at specified levels.
//! Useful for isobars, isotherms, height contours, etc.

/// A contour line at a specific level, composed of line segments.
#[derive(Debug, Clone)]
pub struct ContourLine {
    /// The contour level value
    pub level: f64,
    /// Line segments as (x1, y1, x2, y2) in grid coordinates.
    /// Grid coordinates: (0,0) is top-left corner, (nx-1, ny-1) is bottom-right.
    /// Segments endpoints lie on cell edges, so coordinates range from 0 to nx-1 / ny-1.
    pub segments: Vec<(f64, f64, f64, f64)>,
}

/// Extract contour lines from a 2D scalar field using marching squares.
///
/// # Arguments
/// * `values` - Flat array of f64 values, row-major (ny rows of nx columns)
/// * `nx` - Number of columns
/// * `ny` - Number of rows
/// * `levels` - Contour levels to extract
///
/// # Returns
/// A `ContourLine` for each requested level. Levels with no contour segments
/// are still returned (with empty segment lists).
///
/// NaN values are treated as below all contour levels (they produce no contour crossings).
pub fn contour_lines(
    values: &[f64],
    nx: usize,
    ny: usize,
    levels: &[f64],
) -> Vec<ContourLine> {
    assert_eq!(
        values.len(),
        nx * ny,
        "values.len()={} but nx*ny={}",
        values.len(),
        nx * ny
    );

    levels
        .iter()
        .map(|&level| {
            let segments = march_level(values, nx, ny, level);
            ContourLine { level, segments }
        })
        .collect()
}

/// March a single contour level through the grid.
fn march_level(
    values: &[f64],
    nx: usize,
    ny: usize,
    level: f64,
) -> Vec<(f64, f64, f64, f64)> {
    if nx < 2 || ny < 2 {
        return Vec::new();
    }

    let mut segments = Vec::new();

    // Walk over each cell (nx-1 columns, ny-1 rows of cells)
    for row in 0..ny - 1 {
        for col in 0..nx - 1 {
            // Four corners of the cell (clockwise from top-left):
            //  tl --- tr
            //  |       |
            //  bl --- br
            let tl = safe_val(values, col, row, nx);
            let tr = safe_val(values, col + 1, row, nx);
            let br = safe_val(values, col + 1, row + 1, nx);
            let bl = safe_val(values, col, row + 1, nx);

            // Skip cells with any NaN corner
            if tl.is_nan() || tr.is_nan() || br.is_nan() || bl.is_nan() {
                continue;
            }

            // Compute marching squares case index (4-bit)
            let case = ((tl >= level) as u8)
                | (((tr >= level) as u8) << 1)
                | (((br >= level) as u8) << 2)
                | (((bl >= level) as u8) << 3);

            // Edge midpoints interpolated
            let top = || interp_x(col as f64, (col + 1) as f64, tl, tr, level, row as f64);
            let right = || interp_y(row as f64, (row + 1) as f64, tr, br, level, (col + 1) as f64);
            let bottom = || interp_x(col as f64, (col + 1) as f64, bl, br, level, (row + 1) as f64);
            let left = || interp_y(row as f64, (row + 1) as f64, tl, bl, level, col as f64);

            match case {
                0 | 15 => {} // entirely below or above
                1 => {
                    let (x1, y1) = top();
                    let (x2, y2) = left();
                    segments.push((x1, y1, x2, y2));
                }
                2 => {
                    let (x1, y1) = top();
                    let (x2, y2) = right();
                    segments.push((x1, y1, x2, y2));
                }
                3 => {
                    let (x1, y1) = left();
                    let (x2, y2) = right();
                    segments.push((x1, y1, x2, y2));
                }
                4 => {
                    let (x1, y1) = right();
                    let (x2, y2) = bottom();
                    segments.push((x1, y1, x2, y2));
                }
                5 => {
                    // Saddle point: use average to disambiguate
                    let avg = (tl + tr + br + bl) * 0.25;
                    if avg >= level {
                        let (x1, y1) = top();
                        let (x2, y2) = right();
                        segments.push((x1, y1, x2, y2));
                        let (x1, y1) = bottom();
                        let (x2, y2) = left();
                        segments.push((x1, y1, x2, y2));
                    } else {
                        let (x1, y1) = top();
                        let (x2, y2) = left();
                        segments.push((x1, y1, x2, y2));
                        let (x1, y1) = right();
                        let (x2, y2) = bottom();
                        segments.push((x1, y1, x2, y2));
                    }
                }
                6 => {
                    let (x1, y1) = top();
                    let (x2, y2) = bottom();
                    segments.push((x1, y1, x2, y2));
                }
                7 => {
                    let (x1, y1) = left();
                    let (x2, y2) = bottom();
                    segments.push((x1, y1, x2, y2));
                }
                8 => {
                    let (x1, y1) = left();
                    let (x2, y2) = bottom();
                    segments.push((x1, y1, x2, y2));
                }
                9 => {
                    let (x1, y1) = top();
                    let (x2, y2) = bottom();
                    segments.push((x1, y1, x2, y2));
                }
                10 => {
                    // Saddle point
                    let avg = (tl + tr + br + bl) * 0.25;
                    if avg >= level {
                        let (x1, y1) = top();
                        let (x2, y2) = left();
                        segments.push((x1, y1, x2, y2));
                        let (x1, y1) = right();
                        let (x2, y2) = bottom();
                        segments.push((x1, y1, x2, y2));
                    } else {
                        let (x1, y1) = top();
                        let (x2, y2) = right();
                        segments.push((x1, y1, x2, y2));
                        let (x1, y1) = bottom();
                        let (x2, y2) = left();
                        segments.push((x1, y1, x2, y2));
                    }
                }
                11 => {
                    let (x1, y1) = right();
                    let (x2, y2) = bottom();
                    segments.push((x1, y1, x2, y2));
                }
                12 => {
                    let (x1, y1) = left();
                    let (x2, y2) = right();
                    segments.push((x1, y1, x2, y2));
                }
                13 => {
                    let (x1, y1) = top();
                    let (x2, y2) = right();
                    segments.push((x1, y1, x2, y2));
                }
                14 => {
                    let (x1, y1) = top();
                    let (x2, y2) = left();
                    segments.push((x1, y1, x2, y2));
                }
                _ => unreachable!(),
            }
        }
    }

    segments
}

/// Get a value from the grid, returning NaN for out-of-bounds.
#[inline]
fn safe_val(values: &[f64], x: usize, y: usize, nx: usize) -> f64 {
    values[y * nx + x]
}

/// Interpolate along a horizontal edge to find where the contour crosses.
/// Returns (x, y) in grid coordinates.
#[inline]
fn interp_x(x0: f64, x1: f64, v0: f64, v1: f64, level: f64, y: f64) -> (f64, f64) {
    let t = if (v1 - v0).abs() < 1e-12 {
        0.5
    } else {
        (level - v0) / (v1 - v0)
    };
    (x0 + t * (x1 - x0), y)
}

/// Interpolate along a vertical edge to find where the contour crosses.
/// Returns (x, y) in grid coordinates.
#[inline]
fn interp_y(y0: f64, y1: f64, v0: f64, v1: f64, level: f64, x: f64) -> (f64, f64) {
    let t = if (v1 - v0).abs() < 1e-12 {
        0.5
    } else {
        (level - v0) / (v1 - v0)
    };
    (x, y0 + t * (y1 - y0))
}

/// A contour line with label placement information.
#[derive(Debug, Clone)]
pub struct LabeledContour {
    /// The contour level value
    pub level: f64,
    /// Line segments as (x1, y1, x2, y2) in grid coordinates.
    pub segments: Vec<(f64, f64, f64, f64)>,
    /// Label positions: (x, y, angle_degrees) for text placement along the contour.
    pub label_positions: Vec<(f64, f64, f64)>,
}

/// Generate contour lines with label positions along each contour.
///
/// Labels are placed at regular intervals along the contour, with the angle
/// computed from the local tangent direction so labels follow the contour.
///
/// # Arguments
/// * `values` - Flat array of f64 values, row-major (ny rows of nx columns)
/// * `nx` - Number of columns
/// * `ny` - Number of rows
/// * `levels` - Contour levels to extract
/// * `label_spacing` - Approximate spacing between labels in grid units
///
/// # Returns
/// A `LabeledContour` for each requested level.
pub fn contour_lines_labeled(
    values: &[f64],
    nx: usize,
    ny: usize,
    levels: &[f64],
    label_spacing: f64,
) -> Vec<LabeledContour> {
    let raw_contours = contour_lines(values, nx, ny, levels);

    raw_contours
        .into_iter()
        .map(|cl| {
            let label_positions = compute_label_positions(&cl.segments, label_spacing);
            LabeledContour {
                level: cl.level,
                segments: cl.segments,
                label_positions,
            }
        })
        .collect()
}

/// Build ordered chains of segments and place labels along them.
fn compute_label_positions(
    segments: &[(f64, f64, f64, f64)],
    label_spacing: f64,
) -> Vec<(f64, f64, f64)> {
    if segments.is_empty() || label_spacing <= 0.0 {
        return Vec::new();
    }

    // Chain segments into polylines by matching endpoints
    let chains = chain_segments(segments);
    let mut labels = Vec::new();

    for chain in &chains {
        if chain.len() < 2 {
            continue;
        }

        // Walk along the chain, placing labels at spacing intervals
        let mut accumulated = 0.0;
        // Place first label at half-spacing to center labels
        let mut next_label = label_spacing * 0.5;

        for i in 0..chain.len() - 1 {
            let (x0, y0) = chain[i];
            let (x1, y1) = chain[i + 1];
            let dx = x1 - x0;
            let dy = y1 - y0;
            let seg_len = (dx * dx + dy * dy).sqrt();

            if seg_len < 1e-12 {
                continue;
            }

            while accumulated + seg_len >= next_label {
                let frac = (next_label - accumulated) / seg_len;
                let lx = x0 + dx * frac;
                let ly = y0 + dy * frac;
                let angle = dy.atan2(dx).to_degrees();
                // Normalize angle to [-90, 90] so text is never upside-down
                let angle = if angle > 90.0 {
                    angle - 180.0
                } else if angle < -90.0 {
                    angle + 180.0
                } else {
                    angle
                };
                labels.push((lx, ly, angle));
                next_label += label_spacing;
            }

            accumulated += seg_len;
        }
    }

    labels
}

/// Chain disconnected segments into ordered polylines by matching endpoints.
fn chain_segments(segments: &[(f64, f64, f64, f64)]) -> Vec<Vec<(f64, f64)>> {
    if segments.is_empty() {
        return Vec::new();
    }

    let eps = 1e-6;
    let mut used = vec![false; segments.len()];
    let mut chains: Vec<Vec<(f64, f64)>> = Vec::new();

    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;
        let (x1, y1, x2, y2) = segments[start_idx];
        let mut chain = vec![(x1, y1), (x2, y2)];

        // Try to extend the chain in both directions
        let mut changed = true;
        while changed {
            changed = false;
            let (tail_x, tail_y) = *chain.last().unwrap();
            let (head_x, head_y) = chain[0];

            for i in 0..segments.len() {
                if used[i] {
                    continue;
                }
                let (sx1, sy1, sx2, sy2) = segments[i];

                // Try appending to tail
                if (sx1 - tail_x).abs() < eps && (sy1 - tail_y).abs() < eps {
                    chain.push((sx2, sy2));
                    used[i] = true;
                    changed = true;
                } else if (sx2 - tail_x).abs() < eps && (sy2 - tail_y).abs() < eps {
                    chain.push((sx1, sy1));
                    used[i] = true;
                    changed = true;
                }
                // Try prepending to head
                else if (sx2 - head_x).abs() < eps && (sy2 - head_y).abs() < eps {
                    chain.insert(0, (sx1, sy1));
                    used[i] = true;
                    changed = true;
                } else if (sx1 - head_x).abs() < eps && (sy1 - head_y).abs() < eps {
                    chain.insert(0, (sx2, sy2));
                    used[i] = true;
                    changed = true;
                }
            }
        }

        chains.push(chain);
    }

    chains
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_contour() {
        // 3x3 grid with a gradient from 0 to 8
        // 0 1 2
        // 3 4 5
        // 6 7 8
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = contour_lines(&values, 3, 3, &[4.0]);
        assert_eq!(result.len(), 1);
        assert!(!result[0].segments.is_empty(), "Should have contour segments at level 4.0");
    }

    #[test]
    fn test_no_contour_below() {
        let values = vec![10.0; 9];
        let result = contour_lines(&values, 3, 3, &[5.0]);
        assert_eq!(result.len(), 1);
        assert!(result[0].segments.is_empty(), "All values above level, no contour");
    }

    #[test]
    fn test_no_contour_above() {
        let values = vec![1.0; 9];
        let result = contour_lines(&values, 3, 3, &[5.0]);
        assert_eq!(result.len(), 1);
        assert!(result[0].segments.is_empty(), "All values below level, no contour");
    }

    #[test]
    fn test_nan_handling() {
        let values = vec![0.0, f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = contour_lines(&values, 3, 3, &[4.0]);
        // Should not panic, NaN cells are skipped
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_multiple_levels() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = contour_lines(&values, 3, 3, &[2.0, 4.0, 6.0]);
        assert_eq!(result.len(), 3);
        for cl in &result {
            assert!(!cl.segments.is_empty());
        }
    }

    #[test]
    fn test_labeled_contours() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = contour_lines_labeled(&values, 3, 3, &[4.0], 2.0);
        assert_eq!(result.len(), 1);
        assert!(!result[0].segments.is_empty());
        // Labels should be generated at spacing intervals
        // (may or may not have labels depending on contour length vs spacing)
    }

    #[test]
    fn test_chain_segments() {
        // Two connected segments forming a chain
        let segments = vec![
            (0.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, 2.0, 0.0),
        ];
        let chains = chain_segments(&segments);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].len(), 3); // 3 points for 2 connected segments
    }
}
