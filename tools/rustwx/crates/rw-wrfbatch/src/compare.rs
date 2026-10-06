//! Model-against-reference comparison sheets: the parts that hold without a
//! file.
//!
//! `rw_compare` draws a run's product beside a reference model's own field
//! for the same valid time, on the run's grid, in the run's projection, with
//! one colour scale.  What lives here is everything that decides whether the
//! two panels are COMPARABLE, kept apart from the binary so it can be tested
//! on grids a test can build:
//!
//! * [`match_grids`] -- which reference grid point each run grid point is
//!   compared with.  The answer is a rule, not an assumption: either every
//!   run point's nearest reference point is the same translation (the run is
//!   a window of the reference grid, so the comparison is point against the
//!   same point and nothing is interpolated), or it is nearest-point sampling
//!   and the sheet says so.
//! * [`difference_scale`] -- the diverging ladder of the optional third
//!   panel.  Symmetric, with a neutral bin CENTRED on zero, and stepped in
//!   the units the field is drawn in rather than fitted to the data: a
//!   difference panel whose range moved with every frame could not be read
//!   against the next lead's.
//! * [`sheet_layout`] / [`compose_sheet`] -- the sheet itself: equal panels
//!   in one row under one header band, composed by
//!   `rustwx_render::compose_panel_images`.  Pixels are pasted, never
//!   recomputed, and no panel is resized.
//!
//! Nothing in this module knows which reference model it is being used for.

use rustwx_render::{
    Color, ColorScale, DiscreteColorScale, ExtendMode, PanelGridLayout, PanelPadding, RgbaImage,
    compose_panel_images,
};

/// Kilometres per degree of latitude on the sphere the two grids' own
/// coordinates are compared on.  Only RATIOS of distances decide anything
/// here, so the exact radius does not change a result.
const KM_PER_DEGREE: f64 = 111.195;

/// A run point whose nearest reference point is further than this many
/// reference grid lengths has no reference value: it lies outside the
/// reference domain, and the nearest edge value is not its value.
const OUTSIDE_REFERENCE_SPACINGS: f64 = 0.75;

/// How far, in reference cells, a run point may sit from where the fitted
/// lattice puts it and still be on that lattice.
const LATTICE_TOLERANCE_CELLS: f64 = 0.05;

/// The largest scale difference between two encodings of ONE lattice that
/// is believed to be an encoding difference.  Two earth radii in use for
/// the same grid (6370.000 km in the model, 6371.229 km in its GRIB grid
/// definition) differ by two parts in ten thousand; a part in a thousand
/// is already another grid spacing.
const MAX_ENCODING_SCALE: f64 = 1.0e-3;

/// A slope that moves a point by less than this many cells across the
/// whole run is no slope: the offset is then judged as a constant.
const NO_SLOPE_CELLS: f64 = 0.01;

/// How the run lattice sits on the reference's, measured in reference
/// cells along the reference's own two axes.
///
/// Two encodings of one grid need not agree to the metre.  A GRIB grid
/// definition is a first point, a spacing and an earth radius; a model's
/// own latitude array was computed on the model's radius.  When the two
/// radii differ the decoded reference lattice is the model's lattice
/// SCALED about the first point, so a run on the very same grid sits a
/// growing fraction of a cell from its partners -- 0.4 of a cell at the far
/// corner of a continental grid -- and a distance threshold cannot tell
/// that from a grid that is merely parallel to the reference with its
/// origin shifted, whose values really are the neighbours'.
///
/// The fit can.  In reference index space a scale difference about an
/// anchor is exactly `offset = scale * (index - anchor)` on each axis, so
/// the offsets are regressed on the index and the two cases separate on
/// where the fitted offset is zero: on one lattice the anchor is a point
/// OF the reference grid, and on a shifted one it lies far outside it (or
/// nowhere, when there is no slope at all).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeFit {
    /// Scale difference along the reference's column and row axes.
    pub scale: (f64, f64),
    /// The reference `(column, row)` where the two lattices coincide;
    /// NaN on an axis with no slope.
    pub anchor: (f64, f64),
    /// The largest offset of a run point from its partner, in cells.
    pub max_offset_cells: f64,
    /// The largest departure from the fitted lattice, in cells.
    pub max_residual_cells: f64,
    /// Whether the run's points are the reference's points.
    pub same_lattice: bool,
}

/// How the reference grid was put on the run grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchRule {
    /// Every run point's nearest reference point is the same index offset
    /// AND the two lattices are one ([`LatticeFit::same_lattice`]): the run
    /// grid is a window of the reference grid, `(i0, j0)` being the
    /// reference index of the run's first point.  Point against the same
    /// point; nothing is interpolated.
    ///
    /// The two grids may store their rows or columns in opposite orders
    /// (a decoder that hands back north-first rows against a model that
    /// writes south-first ones); they are still the same points, so the
    /// order is part of the rule rather than a reason to call it
    /// something else.
    Window {
        i0: usize,
        j0: usize,
        rows_reversed: bool,
        columns_reversed: bool,
    },
    /// The grids do not coincide; each run point takes the value of its
    /// nearest reference point.
    Nearest,
}

impl MatchRule {
    /// The sentence a sheet's subtitle and the report carry.
    pub fn describe(self, reference: &str) -> String {
        match self {
            Self::Window { .. } => format!("{reference} at the same grid points"),
            Self::Nearest => format!("{reference} at the nearest grid point"),
        }
    }
}

/// The outcome of [`match_grids`].
#[derive(Debug, Clone)]
pub struct GridMatch {
    pub rule: MatchRule,
    /// For every run point (row-major), the reference flat index it is
    /// compared with, or `u32::MAX` when it lies outside the reference
    /// domain.
    pub source_index: Vec<u32>,
    /// Run points with no reference value.
    pub missing: usize,
    /// The largest run-to-reference distance among matched points.
    pub max_distance_km: f64,
    /// The reference grid length the distances were judged against.
    pub source_spacing_km: f64,
    /// How the run lattice sits on the reference's; `None` when the index
    /// mapping is not a pure translation, so there is no lattice to fit.
    pub lattice: Option<LatticeFit>,
}

fn squared_distance_deg(lat_a: f64, lon_a: f64, lat_b: f64, lon_b: f64, cos_lat: f64) -> f64 {
    let dlat = lat_a - lat_b;
    let mut dlon = lon_a - lon_b;
    if dlon > 180.0 {
        dlon -= 360.0;
    } else if dlon < -180.0 {
        dlon += 360.0;
    }
    let dx = dlon * cos_lat;
    dlat * dlat + dx * dx
}

/// Match every run (target) grid point to a reference (source) grid point.
///
/// No projection is assumed and none is needed: the first point is found by
/// exhaustive search, and every later point descends from its neighbour's
/// answer, which converges on any grid whose coordinates vary smoothly.
/// The rule is then READ OFF the result: a pure index translation between
/// two encodings of one lattice ([`LatticeFit`]) is a window, and anything
/// else is nearest-point sampling.
#[allow(clippy::too_many_arguments)]
pub fn match_grids(
    target_lat: &[f32],
    target_lon: &[f32],
    target_ny: usize,
    target_nx: usize,
    source_lat: &[f32],
    source_lon: &[f32],
    source_ny: usize,
    source_nx: usize,
) -> Result<GridMatch, String> {
    let target_points = target_ny
        .checked_mul(target_nx)
        .ok_or("the run grid dimensions overflow")?;
    let source_points = source_ny
        .checked_mul(source_nx)
        .ok_or("the reference grid dimensions overflow")?;
    if target_points == 0 || source_points == 0 {
        return Err("a grid with no points cannot be compared".into());
    }
    if target_lat.len() != target_points || target_lon.len() != target_points {
        return Err(format!(
            "the {target_ny}x{target_nx} run grid needs {target_points} coordinate pair(s); got lat {} lon {}",
            target_lat.len(),
            target_lon.len()
        ));
    }
    if source_lat.len() != source_points || source_lon.len() != source_points {
        return Err(format!(
            "the {source_ny}x{source_nx} reference grid needs {source_points} coordinate pair(s); got lat {} lon {}",
            source_lat.len(),
            source_lon.len()
        ));
    }
    if source_points >= u32::MAX as usize {
        return Err("the reference grid has too many points to index".into());
    }
    if source_nx < 2 || source_ny < 2 {
        return Err("the reference grid needs at least two points along each axis".into());
    }

    let s_lat = |index: usize| f64::from(source_lat[index]);
    let s_lon = |index: usize| f64::from(source_lon[index]);

    // The reference grid length, from its own first cell.
    let cos0 = s_lat(0).to_radians().cos();
    let along_x = squared_distance_deg(s_lat(0), s_lon(0), s_lat(1), s_lon(1), cos0).sqrt();
    let along_y =
        squared_distance_deg(s_lat(0), s_lon(0), s_lat(source_nx), s_lon(source_nx), cos0).sqrt();
    let spacing_deg = along_x.max(along_y);
    if !spacing_deg.is_finite() || spacing_deg <= 0.0 {
        return Err("the reference grid's first cell has no extent".into());
    }
    let outside_sq = (OUTSIDE_REFERENCE_SPACINGS * spacing_deg).powi(2);

    // Seed: the reference point nearest the run's first point, by search.
    let (t_lat0, t_lon0) = (f64::from(target_lat[0]), f64::from(target_lon[0]));
    let seed_cos = t_lat0.to_radians().cos();
    let mut seed = 0usize;
    let mut seed_distance = f64::INFINITY;
    for index in 0..source_points {
        let distance = squared_distance_deg(t_lat0, t_lon0, s_lat(index), s_lon(index), seed_cos);
        if distance < seed_distance {
            seed_distance = distance;
            seed = index;
        }
    }

    let descend = |lat: f64, lon: f64, start: usize| -> (usize, f64) {
        let cos_lat = lat.to_radians().cos();
        let mut best = start;
        let mut best_distance = squared_distance_deg(lat, lon, s_lat(best), s_lon(best), cos_lat);
        loop {
            let (row, column) = (best / source_nx, best % source_nx);
            let mut improved = false;
            for dj in -1i64..=1 {
                for di in -1i64..=1 {
                    if dj == 0 && di == 0 {
                        continue;
                    }
                    let j = row as i64 + dj;
                    let i = column as i64 + di;
                    if j < 0 || i < 0 || j >= source_ny as i64 || i >= source_nx as i64 {
                        continue;
                    }
                    let index = j as usize * source_nx + i as usize;
                    let distance =
                        squared_distance_deg(lat, lon, s_lat(index), s_lon(index), cos_lat);
                    if distance < best_distance {
                        best_distance = distance;
                        best = index;
                        improved = true;
                    }
                }
            }
            if !improved {
                return (best, best_distance);
            }
        }
    };

    let mut nearest = vec![0usize; target_points];
    let mut source_index = vec![u32::MAX; target_points];
    let mut missing = 0usize;
    let mut max_distance_sq = 0.0f64;
    for j in 0..target_ny {
        for i in 0..target_nx {
            let point = j * target_nx + i;
            // From the neighbour's answer, whichever way the reference
            // stores its rows: the descent takes the one step itself.
            let start = if i > 0 {
                nearest[point - 1]
            } else if j > 0 {
                nearest[point - target_nx]
            } else {
                seed
            };
            let (found, distance) = descend(
                f64::from(target_lat[point]),
                f64::from(target_lon[point]),
                start,
            );
            nearest[point] = found;
            if distance.is_finite() && distance <= outside_sq {
                source_index[point] = found as u32;
                max_distance_sq = max_distance_sq.max(distance);
            } else {
                missing += 1;
            }
        }
    }

    // A window is a pure translation with nothing missing, in whichever
    // order the reference stores its rows and columns.
    let (j0, i0) = (nearest[0] / source_nx, nearest[0] % source_nx);
    let is_window = |rows_reversed: bool, columns_reversed: bool| -> bool {
        let row_of = |j: usize| {
            if rows_reversed {
                j0.checked_sub(j)
            } else {
                Some(j0 + j).filter(|row| *row < source_ny)
            }
        };
        let column_of = |i: usize| {
            if columns_reversed {
                i0.checked_sub(i)
            } else {
                Some(i0 + i).filter(|column| *column < source_nx)
            }
        };
        (0..target_ny).all(|j| {
            let Some(row) = row_of(j) else {
                return false;
            };
            (0..target_nx).all(|i| {
                column_of(i).is_some_and(|column| {
                    nearest[j * target_nx + i] == row * source_nx + column
                })
            })
        })
    };
    let mut rule = MatchRule::Nearest;
    let mut lattice = None;
    if missing == 0 {
        for (rows_reversed, columns_reversed) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            // A one-row or one-column run cannot tell the two orders
            // apart and takes the first that fits.
            if is_window(rows_reversed, columns_reversed) {
                let fit = fit_lattice(
                    target_lat, target_lon, target_ny, target_nx, source_lat, source_lon,
                    source_ny, source_nx, &nearest,
                );
                if fit.is_some_and(|fit| fit.same_lattice) {
                    rule = MatchRule::Window {
                        i0,
                        j0,
                        rows_reversed,
                        columns_reversed,
                    };
                }
                lattice = fit;
                break;
            }
        }
    }

    Ok(GridMatch {
        rule,
        source_index,
        missing,
        max_distance_km: max_distance_sq.sqrt() * KM_PER_DEGREE,
        source_spacing_km: spacing_deg * KM_PER_DEGREE,
        lattice,
    })
}

/// One axis of a [`LatticeFit`]: offsets regressed on the reference index.
struct AxisFit {
    scale: f64,
    anchor: f64,
    max_residual: f64,
    same_lattice: bool,
}

fn fit_axis(samples: &[(f64, f64)], reference_points: usize) -> AxisFit {
    let count = samples.len() as f64;
    let (mut sum_x, mut sum_y, mut sum_xx, mut sum_xy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for (index, offset) in samples {
        sum_x += index;
        sum_y += offset;
        sum_xx += index * index;
        sum_xy += index * offset;
        low = low.min(*index);
        high = high.max(*index);
    }
    let spread = count * sum_xx - sum_x * sum_x;
    let slope = if spread > 0.0 {
        (count * sum_xy - sum_x * sum_y) / spread
    } else {
        0.0
    };
    let intercept = (sum_y - slope * sum_x) / count;
    let max_residual = samples
        .iter()
        .map(|(index, offset)| (offset - (intercept + slope * index)).abs())
        .fold(0.0f64, f64::max);
    let span = (high - low).max(0.0);
    let on_fit = max_residual <= LATTICE_TOLERANCE_CELLS;
    if slope.abs() * span < NO_SLOPE_CELLS {
        // No drift across the run: the two agree only if the constant
        // offset is itself nothing.
        let middle = intercept + slope * 0.5 * (low + high);
        return AxisFit {
            scale: slope,
            anchor: f64::NAN,
            max_residual,
            same_lattice: on_fit && middle.abs() <= LATTICE_TOLERANCE_CELLS,
        };
    }
    let anchor = -intercept / slope;
    AxisFit {
        scale: slope,
        anchor,
        max_residual,
        same_lattice: on_fit
            && slope.abs() <= MAX_ENCODING_SCALE
            && anchor >= -1.0
            && anchor <= reference_points as f64,
    }
}

/// Fit the run lattice to the reference's ([`LatticeFit`]).  `nearest` is
/// the reference flat index of every run point.
#[allow(clippy::too_many_arguments)]
fn fit_lattice(
    target_lat: &[f32],
    target_lon: &[f32],
    target_ny: usize,
    target_nx: usize,
    source_lat: &[f32],
    source_lon: &[f32],
    source_ny: usize,
    source_nx: usize,
    nearest: &[usize],
) -> Option<LatticeFit> {
    // East and north, in degrees of latitude, from reference point `from`
    // to the point `(lat, lon)`.
    let vector = |from: usize, lat: f64, lon: f64| -> (f64, f64) {
        let from_lat = f64::from(source_lat[from]);
        let mut dlon = lon - f64::from(source_lon[from]);
        if dlon > 180.0 {
            dlon -= 360.0;
        } else if dlon < -180.0 {
            dlon += 360.0;
        }
        (dlon * from_lat.to_radians().cos(), lat - from_lat)
    };
    let stride_j = (target_ny / 200).max(1);
    let stride_i = (target_nx / 200).max(1);
    let mut columns: Vec<(f64, f64)> = Vec::new();
    let mut rows: Vec<(f64, f64)> = Vec::new();
    let mut max_offset = 0.0f64;
    for j in (0..target_ny).step_by(stride_j) {
        for i in (0..target_nx).step_by(stride_i) {
            let point = j * target_nx + i;
            let partner = nearest[point];
            let (row, column) = (partner / source_nx, partner % source_nx);
            // The reference's own axes at the partner, one cell long.
            let along_columns = if column + 1 < source_nx {
                let next = partner + 1;
                vector(partner, f64::from(source_lat[next]), f64::from(source_lon[next]))
            } else {
                let previous = partner - 1;
                let back =
                    vector(partner, f64::from(source_lat[previous]), f64::from(source_lon[previous]));
                (-back.0, -back.1)
            };
            let along_rows = if row + 1 < source_ny {
                let next = partner + source_nx;
                vector(partner, f64::from(source_lat[next]), f64::from(source_lon[next]))
            } else {
                let previous = partner - source_nx;
                let back =
                    vector(partner, f64::from(source_lat[previous]), f64::from(source_lon[previous]));
                (-back.0, -back.1)
            };
            let offset = vector(
                partner,
                f64::from(target_lat[point]),
                f64::from(target_lon[point]),
            );
            let determinant = along_columns.0 * along_rows.1 - along_columns.1 * along_rows.0;
            if !determinant.is_finite() || determinant == 0.0 {
                return None;
            }
            let in_columns = (offset.0 * along_rows.1 - offset.1 * along_rows.0) / determinant;
            let in_rows = (along_columns.0 * offset.1 - along_columns.1 * offset.0) / determinant;
            if !in_columns.is_finite() || !in_rows.is_finite() {
                return None;
            }
            max_offset = max_offset.max(in_columns.abs()).max(in_rows.abs());
            columns.push((column as f64, in_columns));
            rows.push((row as f64, in_rows));
        }
    }
    if columns.is_empty() {
        return None;
    }
    let column_fit = fit_axis(&columns, source_nx);
    let row_fit = fit_axis(&rows, source_ny);
    Some(LatticeFit {
        scale: (column_fit.scale, row_fit.scale),
        anchor: (column_fit.anchor, row_fit.anchor),
        max_offset_cells: max_offset,
        max_residual_cells: column_fit.max_residual.max(row_fit.max_residual),
        same_lattice: column_fit.same_lattice && row_fit.same_lattice,
    })
}

/// The reference plane on the run grid.  Unmatched points are NaN, which
/// the renderer leaves undrawn.
pub fn sample(source_values: &[f32], grid_match: &GridMatch) -> Vec<f32> {
    grid_match
        .source_index
        .iter()
        .map(|index| {
            if *index == u32::MAX {
                f32::NAN
            } else {
                source_values
                    .get(*index as usize)
                    .copied()
                    .unwrap_or(f32::NAN)
            }
        })
        .collect()
}

/// `left - right`, NaN wherever either side is.
pub fn difference(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            if a.is_finite() && b.is_finite() {
                a - b
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// `left - right` for a field whose scale leaves everything below `floor`
/// undrawn (reflectivity under its first step, a trace of rain).
///
/// Two models rarely spell "nothing here" with the same number: one
/// writes -35 dBZ for no echo and the other -10, so a plain difference is
/// -25 over every clear-sky point and the panel is one colour that means
/// nothing.  Below the floor neither panel draws anything, so:
///
/// * where BOTH sides are below it, there is no difference to draw (NaN);
/// * where one side is below it, that side counts as the floor, and the
///   difference is how far the other side stands above what is drawn.
///
/// With no floor this is [`difference`].
pub fn difference_above_floor(left: &[f32], right: &[f32], floor: Option<f64>) -> Vec<f32> {
    let Some(floor) = floor.filter(|floor| floor.is_finite()) else {
        return difference(left, right);
    };
    let floor = floor as f32;
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            if !a.is_finite() || !b.is_finite() || (*a < floor && *b < floor) {
                f32::NAN
            } else {
                a.max(floor) - b.max(floor)
            }
        })
        .collect()
}

/// Speed from two wind components.  Speed is the same number in grid-
/// relative and earth-relative components, so no rotation is involved.
pub fn wind_speed(u: &[f32], v: &[f32]) -> Vec<f32> {
    u.iter()
        .zip(v)
        .map(|(u, v)| {
            if u.is_finite() && v.is_finite() {
                u.hypot(*v)
            } else {
                f32::NAN
            }
        })
        .collect()
}

/// What a difference plane amounts to, for the event line and the sidecar.
/// Never drawn on the sheet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferenceStats {
    pub points: usize,
    pub mean: f64,
    pub rms: f64,
    pub max_abs: f64,
}

pub fn difference_stats(values: &[f32]) -> Option<DifferenceStats> {
    let mut points = 0usize;
    let (mut sum, mut sum_sq, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        let value = f64::from(value);
        points += 1;
        sum += value;
        sum_sq += value * value;
        max_abs = max_abs.max(value.abs());
    }
    (points > 0).then(|| DifferenceStats {
        points,
        mean: sum / points as f64,
        rms: (sum_sq / points as f64).sqrt(),
        max_abs,
    })
}

/// Blue (run below reference) through a neutral centre to red (run above).
const DIFFERENCE_COLORS: [[u8; 3]; 11] = [
    [5, 48, 97],
    [33, 102, 172],
    [67, 147, 195],
    [146, 197, 222],
    [209, 229, 240],
    [247, 247, 247],
    [253, 219, 199],
    [244, 165, 130],
    [214, 96, 77],
    [178, 24, 43],
    [103, 0, 31],
];

/// The width of one difference bin, in the units the field is DRAWN in.
///
/// Keyed on the display units and nothing else, so a field added to the
/// product table needs no row here unless its units are new.  `None` for
/// units with no agreed step: the difference panel is then not drawn,
/// rather than drawn on a ladder invented for the occasion.
pub fn difference_step(display_units: &str) -> Option<f64> {
    let units = display_units.trim().to_ascii_lowercase();
    match units.as_str() {
        "degf" | "f" | "\u{b0}f" => Some(2.0),
        "degc" | "c" | "\u{b0}c" | "k" => Some(1.0),
        "kt" | "kts" | "knots" => Some(2.0),
        "m/s" | "m s-1" | "m s^-1" => Some(1.0),
        "dam" => Some(0.5),
        "m" | "gpm" => Some(10.0),
        "dbz" => Some(5.0),
        "in" | "inch" | "inches" => Some(0.1),
        "mm" | "kg/m^2" | "kg m-2" => Some(2.0),
        "hpa" | "mb" => Some(1.0),
        "w/m^2" | "w/m2" | "w m-2" | "w m^-2" | "w m**-2" => Some(10.0),
        _ => None,
    }
}

/// The diverging ladder of a difference panel: eleven bins of width `step`,
/// the middle one centred on zero, open at both ends.
///
/// An ODD number of bins on purpose.  With an even number zero is a colour
/// break, and a field that agrees to within rounding is painted half pale
/// blue and half pale red, which reads as a pattern.
pub fn difference_scale(step: f64) -> ColorScale {
    let step = if step.is_finite() && step > 0.0 {
        step
    } else {
        1.0
    };
    let bins = DIFFERENCE_COLORS.len();
    let half = bins as f64 / 2.0;
    let levels: Vec<f64> = (0..=bins)
        .map(|index| (index as f64 - half) * step)
        .collect();
    ColorScale::Discrete(DiscreteColorScale {
        levels,
        colors: DIFFERENCE_COLORS
            .iter()
            .map(|[r, g, b]| Color::rgba(*r, *g, *b, 255))
            .collect(),
        extend: ExtendMode::Both,
        mask_below: None,
    })
}

/// A fixed filled ladder: `bins` steps of `step` from `first`, coloured by
/// reading `palette` end to end.
///
/// For a field the production catalogue only ever contours, so that no
/// production fill exists to borrow.  The range is the caller's and never
/// the data's: two panels drawn to be compared, and the next lead's two
/// after them, have to mean the same thing by the same colour.
pub fn ladder_scale(first: f64, step: f64, bins: usize, palette: &[Color]) -> ColorScale {
    let step = if step.is_finite() && step > 0.0 {
        step
    } else {
        1.0
    };
    let first = if first.is_finite() { first } else { 0.0 };
    let bins = bins.max(2);
    let levels: Vec<f64> = (0..=bins).map(|index| first + index as f64 * step).collect();
    let colors: Vec<Color> = (0..bins)
        .map(|index| {
            if palette.is_empty() {
                return Color::BLACK;
            }
            let position = index as f64 / (bins - 1) as f64 * (palette.len() - 1) as f64;
            let lower = position.floor() as usize;
            let upper = (lower + 1).min(palette.len() - 1);
            let weight = position - lower as f64;
            let blend = |a: u8, b: u8| {
                (f64::from(a) + (f64::from(b) - f64::from(a)) * weight).round() as u8
            };
            Color::rgba(
                blend(palette[lower].r, palette[upper].r),
                blend(palette[lower].g, palette[upper].g),
                blend(palette[lower].b, palette[upper].b),
                255,
            )
        })
        .collect();
    ColorScale::Discrete(DiscreteColorScale {
        levels,
        colors,
        extend: ExtendMode::Both,
        mask_below: None,
    })
}

/// The header band and gutters of a sheet, sized from its panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SheetMetrics {
    pub header_height: u32,
    pub gap: u32,
    pub margin_x: u32,
    pub title_scale: u32,
    pub subtitle_scale: u32,
    pub title_y: u32,
    pub subtitle_y: u32,
}

/// Header metrics for panels `panel_width` wide.
///
/// Everything follows the panel width so a 2400-pixel panel does not get a
/// 1200-pixel panel's header.  The pixel sizes are the renderer's own
/// (`rustwx_render` draws bold text at `15 + 4 (scale - 1)` px and regular
/// at `12 + 4 (scale - 1)` px), and the band is tall enough for both lines
/// with air above, between and below them.
pub fn sheet_metrics(panel_width: u32) -> SheetMetrics {
    let title_scale = (panel_width as f64 / 240.0).round().clamp(2.0, 9.0) as u32;
    let subtitle_scale = title_scale.saturating_sub(1).max(1);
    let title_px = 15 + 4 * (title_scale - 1);
    let subtitle_px = 12 + 4 * (subtitle_scale - 1);
    let pad = (title_px / 2).max(8);
    let title_y = pad;
    let subtitle_y = title_y + title_px + pad / 2 + 2;
    let header_height = subtitle_y + subtitle_px + pad;
    SheetMetrics {
        header_height,
        gap: (panel_width / 150).max(2),
        margin_x: (panel_width as f64 * 0.032).round() as u32,
        title_scale,
        subtitle_scale,
        title_y,
        subtitle_y,
    }
}

/// One row of `panels` equal panels under one header band.
pub fn sheet_layout(
    panels: usize,
    panel_width: u32,
    panel_height: u32,
    background: Color,
) -> Result<PanelGridLayout, String> {
    if panels == 0 {
        return Err("an empty panel row has no image to compose".to_string());
    }
    let columns = u32::try_from(panels).map_err(|_| "panel count exceeds image dimensions")?;
    let metrics = sheet_metrics(panel_width);
    Ok(
        PanelGridLayout::new(1, columns, panel_width, panel_height)
            .map_err(|error| error.to_string())?
            .with_gaps(metrics.gap, 0)
            .with_padding(PanelPadding {
                top: metrics.header_height,
                right: 0,
                bottom: 0,
                left: 0,
            })
            .with_background(background),
    )
}

/// The text of a sheet's header band.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SheetHeader {
    /// The product and its units: `2 m temperature (degF)`.
    pub title: String,
    /// The facts both panels share: cycle, lead, valid time.
    pub subtitle: String,
}

fn ink_for(background: Color) -> (rustwx_render::Rgba, rustwx_render::Rgba) {
    let luminance = 0.2126 * f64::from(background.r)
        + 0.7152 * f64::from(background.g)
        + 0.0722 * f64::from(background.b);
    if luminance >= 128.0 {
        (
            rustwx_render::Rgba::new(17, 24, 39),
            rustwx_render::Rgba::new(75, 85, 99),
        )
    } else {
        (
            rustwx_render::Rgba::new(243, 244, 246),
            rustwx_render::Rgba::new(176, 184, 196),
        )
    }
}

/// Compose finished panels into one sheet and write the header band.
///
/// The panels must already be the same size: a panel is never resized here,
/// because a resampled map is no longer the map the renderer drew.  The
/// sheet's background is the panels' own (their first pixel), so the band
/// and the gutters belong to whatever theme drew the panels.
pub fn compose_sheet(panels: &[RgbaImage], header: &SheetHeader) -> Result<RgbaImage, String> {
    let first = panels.first().ok_or("a sheet needs at least one panel")?;
    let (width, height) = (first.width(), first.height());
    for (index, panel) in panels.iter().enumerate() {
        if panel.width() != width || panel.height() != height {
            return Err(format!(
                "panel {index} is {}x{} and panel 0 is {width}x{height}; panels of two sizes are not composed, because one of them would have to be resampled",
                panel.width(),
                panel.height()
            ));
        }
    }
    let corner = first.get_pixel(0, 0).0;
    let background = Color::rgba(corner[0], corner[1], corner[2], 255);
    let layout = sheet_layout(panels.len(), width, height, background)?;
    let mut sheet = compose_panel_images(&layout, panels).map_err(|error| error.to_string())?;
    let metrics = sheet_metrics(width);
    let (title_ink, subtitle_ink) = ink_for(background);
    rustwx_render::draw_text_bold(
        &mut sheet,
        &header.title,
        metrics.margin_x as i32,
        metrics.title_y as i32,
        title_ink,
        metrics.title_scale,
    );
    rustwx_render::draw_text(
        &mut sheet,
        &header.subtitle,
        metrics.margin_x as i32,
        metrics.subtitle_y as i32,
        subtitle_ink,
        metrics.subtitle_scale,
    );
    Ok(sheet)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where fractional index `(j, i)` of the test lattice is: rows bend
    /// poleward away from the centre column and columns converge with
    /// latitude, as a conic grid's do.
    fn position(j: f64, i: f64, lat0: f64, lon0: f64, step_deg: f64) -> (f32, f32) {
        let x = i * step_deg;
        let y = j * step_deg;
        let bend = 0.002 * (x - 3.0).powi(2);
        let row_lat = lat0 + y + bend;
        (row_lat as f32, (lon0 + x / row_lat.to_radians().cos()) as f32)
    }

    fn grid(ny: usize, nx: usize, lat0: f64, lon0: f64, step_deg: f64) -> (Vec<f32>, Vec<f32>) {
        grid_at(ny, nx, lat0, lon0, step_deg, |j, i| (j as f64, i as f64))
    }

    /// A grid whose point `(j, i)` sits at the fractional lattice index
    /// `place(j, i)` -- how a scaled or shifted copy of a lattice is made.
    fn grid_at(
        ny: usize,
        nx: usize,
        lat0: f64,
        lon0: f64,
        step_deg: f64,
        place: impl Fn(usize, usize) -> (f64, f64),
    ) -> (Vec<f32>, Vec<f32>) {
        let mut lat = Vec::with_capacity(ny * nx);
        let mut lon = Vec::with_capacity(ny * nx);
        for j in 0..ny {
            for i in 0..nx {
                let (fj, fi) = place(j, i);
                let (point_lat, point_lon) = position(fj, fi, lat0, lon0, step_deg);
                lat.push(point_lat);
                lon.push(point_lon);
            }
        }
        (lat, lon)
    }

    fn window(
        lat: &[f32],
        lon: &[f32],
        nx: usize,
        j0: usize,
        i0: usize,
        ny_w: usize,
        nx_w: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut out_lat = Vec::new();
        let mut out_lon = Vec::new();
        for j in 0..ny_w {
            for i in 0..nx_w {
                out_lat.push(lat[(j0 + j) * nx + i0 + i]);
                out_lon.push(lon[(j0 + j) * nx + i0 + i]);
            }
        }
        (out_lat, out_lon)
    }

    fn discrete(scale: &ColorScale) -> DiscreteColorScale {
        match scale {
            ColorScale::Discrete(scale) => scale.clone(),
            ColorScale::Weather(preset) => preset.scale(),
        }
    }

    #[test]
    fn a_window_of_the_reference_grid_is_matched_point_for_point() {
        let (lat, lon) = grid(40, 60, 30.0, -105.0, 0.03);
        let (w_lat, w_lon) = window(&lat, &lon, 60, 1, 1, 38, 58);
        let found = match_grids(&w_lat, &w_lon, 38, 58, &lat, &lon, 40, 60).expect("match");
        assert_eq!(
            found.rule,
            MatchRule::Window {
                i0: 1,
                j0: 1,
                rows_reversed: false,
                columns_reversed: false
            }
        );
        assert_eq!(found.missing, 0);
        assert!(found.max_distance_km < 1.0e-3, "{}", found.max_distance_km);
        for j in 0..38 {
            for i in 0..58 {
                assert_eq!(found.source_index[j * 58 + i] as usize, (j + 1) * 60 + i + 1);
            }
        }
    }

    #[test]
    fn one_lattice_encoded_on_two_earth_radii_is_still_a_window() {
        // The reference decoded on a slightly different sphere is the run's
        // lattice scaled about its first point: partners drift apart to
        // 0.3 of a cell at the far corner and are the same grid points.
        let scale = 8.0e-4;
        let (lat, lon) = grid(300, 400, 30.0, -105.0, 0.01);
        let (w_lat, w_lon) = grid_at(280, 380, 30.0, -105.0, 0.01, |j, i| {
            ((3 + j) as f64 * (1.0 + scale), (5 + i) as f64 * (1.0 + scale))
        });
        let found = match_grids(&w_lat, &w_lon, 280, 380, &lat, &lon, 300, 400).expect("match");
        assert_eq!(
            found.rule,
            MatchRule::Window {
                i0: 5,
                j0: 3,
                rows_reversed: false,
                columns_reversed: false
            }
        );
        let fit = found.lattice.expect("a translation is fitted");
        assert!(fit.same_lattice);
        assert!(fit.max_offset_cells > 0.25, "{}", fit.max_offset_cells);
        assert!((fit.scale.0 - scale).abs() < 5.0e-5, "{:?}", fit.scale);
        assert!((fit.scale.1 - scale).abs() < 5.0e-5, "{:?}", fit.scale);
        assert!(fit.anchor.0.abs() < 20.0 && fit.anchor.1.abs() < 20.0, "{:?}", fit.anchor);
        assert!(fit.max_residual_cells < LATTICE_TOLERANCE_CELLS);
    }

    #[test]
    fn a_window_anchored_inside_the_reference_grid_is_still_a_window() {
        // A cut whose centre was read off the decoded reference coincides
        // with it THERE and drifts away from it elsewhere.
        let scale = 8.0e-4;
        let (lat, lon) = grid(300, 400, 30.0, -105.0, 0.01);
        let (w_lat, w_lon) = grid_at(200, 300, 30.0, -105.0, 0.01, |j, i| {
            (
                150.0 + ((50 + j) as f64 - 150.0) * (1.0 + scale),
                200.0 + ((50 + i) as f64 - 200.0) * (1.0 + scale),
            )
        });
        let found = match_grids(&w_lat, &w_lon, 200, 300, &lat, &lon, 300, 400).expect("match");
        assert!(matches!(found.rule, MatchRule::Window { i0: 50, j0: 50, .. }));
        let fit = found.lattice.expect("fitted");
        assert!((fit.anchor.0 - 200.0).abs() < 20.0 && (fit.anchor.1 - 150.0).abs() < 20.0);
    }

    #[test]
    fn a_parallel_grid_shifted_by_part_of_a_cell_is_not_the_same_points() {
        // Same spacing, same orientation, origin 0.4 of a cell away: every
        // nearest index is the same translation, and every value would be
        // a neighbour's.  That is nearest-point sampling and is called so.
        let (lat, lon) = grid(40, 60, 30.0, -105.0, 0.03);
        let (w_lat, w_lon) = grid_at(30, 40, 30.0, -105.0, 0.03, |j, i| {
            ((3 + j) as f64 + 0.4, (5 + i) as f64)
        });
        let found = match_grids(&w_lat, &w_lon, 30, 40, &lat, &lon, 40, 60).expect("match");
        assert_eq!(found.rule, MatchRule::Nearest);
        assert_eq!(found.missing, 0);
        let fit = found.lattice.expect("a translation is still fitted");
        assert!(!fit.same_lattice);
        assert!((fit.max_offset_cells - 0.4).abs() < 0.02, "{}", fit.max_offset_cells);
        // The sampling itself is unchanged by what it is called.
        assert_eq!(found.source_index[0] as usize, 3 * 60 + 5);
    }

    #[test]
    fn a_shifted_grid_on_another_radius_is_not_mistaken_for_the_same_lattice() {
        // Shifted AND scaled: the offsets do have a slope, but the place
        // they vanish is far outside the reference grid, so no point of the
        // reference is a point of the run.
        let scale = 2.0e-4;
        let (lat, lon) = grid(300, 400, 30.0, -105.0, 0.01);
        let (w_lat, w_lon) = grid_at(280, 380, 30.0, -105.0, 0.01, |j, i| {
            (
                -1500.0 + ((3 + j) as f64 + 1500.0) * (1.0 + scale),
                -1500.0 + ((5 + i) as f64 + 1500.0) * (1.0 + scale),
            )
        });
        let found = match_grids(&w_lat, &w_lon, 280, 380, &lat, &lon, 300, 400).expect("match");
        assert_eq!(found.rule, MatchRule::Nearest);
        let fit = found.lattice.expect("fitted");
        assert!(fit.anchor.0 < -1000.0 && fit.anchor.1 < -1000.0, "{:?}", fit.anchor);
    }

    #[test]
    fn two_grid_spacings_are_not_one_lattice_however_they_line_up() {
        // One per cent is a different grid, not a different sphere.
        let (lat, lon) = grid(60, 60, 30.0, -105.0, 0.03);
        let (w_lat, w_lon) = grid_at(30, 30, 30.0, -105.0, 0.03, |j, i| {
            ((5 + j) as f64 * 1.01, (5 + i) as f64 * 1.01)
        });
        let found = match_grids(&w_lat, &w_lon, 30, 30, &lat, &lon, 60, 60).expect("match");
        assert_eq!(found.rule, MatchRule::Nearest);
    }

    #[test]
    fn a_reference_stored_north_first_is_still_the_same_grid_points() {
        let (lat, lon) = grid(40, 60, 30.0, -105.0, 0.03);
        // The same reference with its rows written in the opposite order.
        let mut flipped_lat = Vec::with_capacity(lat.len());
        let mut flipped_lon = Vec::with_capacity(lon.len());
        for j in (0..40).rev() {
            flipped_lat.extend_from_slice(&lat[j * 60..(j + 1) * 60]);
            flipped_lon.extend_from_slice(&lon[j * 60..(j + 1) * 60]);
        }
        let (w_lat, w_lon) = window(&lat, &lon, 60, 2, 4, 30, 50);
        let found =
            match_grids(&w_lat, &w_lon, 30, 50, &flipped_lat, &flipped_lon, 40, 60).expect("match");
        assert_eq!(
            found.rule,
            MatchRule::Window {
                i0: 4,
                j0: 37,
                rows_reversed: true,
                columns_reversed: false
            }
        );
        // Values follow the points, not the storage order.
        let values: Vec<f32> = (0..2400).map(|index| index as f32).collect();
        let mut flipped_values = Vec::with_capacity(values.len());
        for j in (0..40).rev() {
            flipped_values.extend_from_slice(&values[j * 60..(j + 1) * 60]);
        }
        let sampled = sample(&flipped_values, &found);
        for j in 0..30 {
            for i in 0..50 {
                assert_eq!(sampled[j * 50 + i], values[(j + 2) * 60 + i + 4]);
            }
        }
    }

    #[test]
    fn a_grid_that_is_not_a_window_is_sampled_at_the_nearest_point_and_says_so() {
        let (lat, lon) = grid(40, 60, 30.0, -105.0, 0.03);
        // Twice the spacing: every second reference point, so no single
        // translation maps run indices to reference indices.
        let (t_lat, t_lon) = grid(15, 20, 30.03, -104.95, 0.06);
        let found = match_grids(&t_lat, &t_lon, 15, 20, &lat, &lon, 40, 60).expect("match");
        assert_eq!(found.rule, MatchRule::Nearest);
        assert!(found.max_distance_km <= 0.75 * found.source_spacing_km + 1.0e-6);
        // Each chosen point really is the nearest one.
        for point in [0usize, 7, 150, 299] {
            let chosen = found.source_index[point];
            if chosen == u32::MAX {
                continue;
            }
            let cos_lat = f64::from(t_lat[point]).to_radians().cos();
            let chosen_distance = squared_distance_deg(
                f64::from(t_lat[point]),
                f64::from(t_lon[point]),
                f64::from(lat[chosen as usize]),
                f64::from(lon[chosen as usize]),
                cos_lat,
            );
            let best = (0..lat.len())
                .map(|index| {
                    squared_distance_deg(
                        f64::from(t_lat[point]),
                        f64::from(t_lon[point]),
                        f64::from(lat[index]),
                        f64::from(lon[index]),
                        cos_lat,
                    )
                })
                .fold(f64::INFINITY, f64::min);
            assert!((chosen_distance - best).abs() < 1.0e-12, "point {point}");
        }
    }

    #[test]
    fn a_run_point_outside_the_reference_domain_has_no_reference_value() {
        let (lat, lon) = grid(20, 20, 30.0, -105.0, 0.03);
        // Starts inside and runs off the east edge.
        let (t_lat, t_lon) = grid(10, 30, 30.09, -104.9, 0.03);
        let found = match_grids(&t_lat, &t_lon, 10, 30, &lat, &lon, 20, 20).expect("match");
        assert_eq!(found.rule, MatchRule::Nearest);
        assert!(found.missing > 0);
        let values: Vec<f32> = (0..lat.len()).map(|index| index as f32).collect();
        let sampled = sample(&values, &found);
        assert_eq!(
            sampled.iter().filter(|value| value.is_nan()).count(),
            found.missing
        );
        assert!(sampled[0].is_finite(), "the first point is inside");
        assert!(sampled[29].is_nan(), "the east end of the first row is outside");
    }

    #[test]
    fn mismatched_coordinate_arrays_are_refused_by_name() {
        let (lat, lon) = grid(4, 4, 30.0, -105.0, 0.03);
        let error = match_grids(&lat[..15], &lon, 4, 4, &lat, &lon, 4, 4).expect_err("short");
        assert!(error.contains("run grid"), "{error}");
        let error = match_grids(&lat, &lon, 4, 4, &lat[..15], &lon, 4, 4).expect_err("short");
        assert!(error.contains("reference grid"), "{error}");
    }

    #[test]
    fn the_difference_ladder_is_symmetric_with_a_neutral_bin_centred_on_zero() {
        let scale = discrete(&difference_scale(2.0));
        assert_eq!(scale.levels.len(), scale.colors.len() + 1);
        assert_eq!(scale.colors.len() % 2, 1, "an odd number of bins");
        assert!(scale.levels.windows(2).all(|pair| pair[1] > pair[0]));
        for (low, high) in scale.levels.iter().zip(scale.levels.iter().rev()) {
            assert!((low + high).abs() < 1.0e-12, "{low} and {high} are not mirror images");
        }
        let middle = scale.colors.len() / 2;
        assert_eq!(scale.levels[middle], -1.0);
        assert_eq!(scale.levels[middle + 1], 1.0);
        let neutral = scale.colors[middle];
        assert_eq!((neutral.r, neutral.g, neutral.b), (247, 247, 247));
        assert_eq!(scale.extend, ExtendMode::Both);
        assert_eq!(scale.mask_below, None, "a negative difference is not masked");
        // Fixed by the step, never by the data.
        assert_eq!(discrete(&difference_scale(2.0)).levels, scale.levels);
        assert_eq!(scale.levels.first().copied(), Some(-11.0));
        assert_eq!(scale.levels.last().copied(), Some(11.0));
    }

    #[test]
    fn a_fixed_ladder_has_one_colour_per_step_and_reads_the_palette_end_to_end() {
        let palette = [
            Color::rgba(0, 0, 0, 255),
            Color::rgba(100, 100, 100, 255),
            Color::rgba(200, 200, 200, 255),
        ];
        let scale = discrete(&ladder_scale(480.0, 6.0, 21, &palette));
        assert_eq!(scale.levels.len(), scale.colors.len() + 1);
        assert_eq!(scale.levels.first().copied(), Some(480.0));
        assert_eq!(scale.levels.last().copied(), Some(606.0));
        assert!(scale.levels.windows(2).all(|pair| (pair[1] - pair[0] - 6.0).abs() < 1.0e-9));
        assert_eq!(scale.colors.first().map(|c| c.r), Some(0));
        assert_eq!(scale.colors[10].r, 100, "the middle step is the middle anchor");
        assert_eq!(scale.colors.last().map(|c| c.r), Some(200));
        // Fixed by the arguments: nothing about the data enters.
        assert_eq!(discrete(&ladder_scale(480.0, 6.0, 21, &palette)).levels, scale.levels);
        // Degenerate requests still make a well-formed ladder.
        for (step, bins) in [(0.0, 0usize), (f64::NAN, 1), (-2.0, 5)] {
            let scale = discrete(&ladder_scale(0.0, step, bins, &[]));
            assert_eq!(scale.levels.len(), scale.colors.len() + 1);
            assert!(scale.levels.windows(2).all(|pair| pair[1] > pair[0]));
        }
    }

    #[test]
    fn a_degenerate_step_does_not_collapse_the_ladder() {
        for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let scale = discrete(&difference_scale(step));
            assert!(scale.levels.windows(2).all(|pair| pair[1] > pair[0]));
        }
    }

    #[test]
    fn difference_steps_follow_the_drawn_units() {
        assert_eq!(difference_step("degF"), Some(2.0));
        assert_eq!(difference_step("kt"), Some(2.0));
        assert_eq!(difference_step("dam"), Some(0.5));
        assert_eq!(difference_step(" m "), Some(10.0));
        assert_eq!(difference_step("W m-2"), Some(10.0));
        assert_eq!(difference_step("W/m^2"), Some(10.0));
        assert_eq!(difference_step("furlongs"), None);
    }

    #[test]
    fn a_difference_is_missing_wherever_either_side_is() {
        let left = [1.0f32, f32::NAN, 3.0, 4.0];
        let right = [0.5f32, 1.0, f32::NAN, 6.0];
        let diff = difference(&left, &right);
        assert_eq!(diff[0], 0.5);
        assert!(diff[1].is_nan() && diff[2].is_nan());
        assert_eq!(diff[3], -2.0);
        let stats = difference_stats(&diff).expect("two finite points");
        assert_eq!(stats.points, 2);
        assert!((stats.mean + 0.75).abs() < 1.0e-9);
        assert!((stats.max_abs - 2.0).abs() < 1.0e-9);
        assert_eq!(difference_stats(&[f32::NAN]), None);
    }

    #[test]
    fn two_spellings_of_nothing_are_not_a_difference() {
        // No echo is -35 on one side and -10 on the other; the scale draws
        // nothing below 5.
        let run = [-35.0f32, 40.0, -35.0, 30.0, f32::NAN];
        let reference = [-10.0f32, -10.0, 25.0, 45.0, 20.0];
        let plain = difference(&run, &reference);
        assert_eq!(plain[0], -25.0, "the floors alone make a difference");
        let drawn = difference_above_floor(&run, &reference, Some(5.0));
        assert!(drawn[0].is_nan(), "nothing on either side is nothing to draw");
        assert_eq!(drawn[1], 35.0, "an echo against none stands 35 above the floor");
        assert_eq!(drawn[2], -20.0);
        assert_eq!(drawn[3], -15.0, "two echoes differ as they are");
        assert!(drawn[4].is_nan());
        // No floor, or none that is a number, is the plain difference.
        assert_eq!(difference_above_floor(&run[..4], &reference[..4], None), plain[..4]);
        assert_eq!(
            difference_above_floor(&run[..4], &reference[..4], Some(f64::NAN)),
            plain[..4]
        );
    }

    #[test]
    fn wind_speed_needs_both_components() {
        let speed = wind_speed(&[3.0, f32::NAN], &[4.0, 1.0]);
        assert_eq!(speed[0], 5.0);
        assert!(speed[1].is_nan());
    }

    #[test]
    fn a_sheet_is_equal_panels_in_one_row_under_one_header() {
        let metrics = sheet_metrics(1200);
        for panels in [1usize, 2, 3, 4, 7, 17] {
            let layout = sheet_layout(panels, 1200, 900, Color::WHITE).expect("layout");
            let (width, height) = layout.canvas_size().expect("canvas");
            assert_eq!(
                width,
                panels as u32 * 1200 + (panels as u32 - 1) * metrics.gap
            );
            assert_eq!(height, 900 + metrics.header_height);
            for index in 0..panels {
                let (x, y) = layout.panel_origin(index).expect("origin");
                assert_eq!(x, index as u32 * (1200 + metrics.gap));
                assert_eq!(y, metrics.header_height, "every panel starts under the header");
            }
        }
        assert!(sheet_layout(0, 1200, 900, Color::WHITE).is_err());
    }

    #[test]
    fn the_header_band_holds_both_lines_at_every_panel_width() {
        for width in [600u32, 1200, 1800, 2400] {
            let metrics = sheet_metrics(width);
            let title_px = 15 + 4 * (metrics.title_scale - 1);
            let subtitle_px = 12 + 4 * (metrics.subtitle_scale - 1);
            assert!(metrics.title_y + title_px <= metrics.subtitle_y, "width {width}");
            assert!(
                metrics.subtitle_y + subtitle_px <= metrics.header_height,
                "width {width}"
            );
        }
        assert!(sheet_metrics(2400).header_height > sheet_metrics(1200).header_height);
    }

    #[test]
    fn panels_are_pasted_unchanged_and_never_resized() {
        let left = RgbaImage::from_pixel(64, 48, image::Rgba([10, 20, 30, 255]));
        let right = RgbaImage::from_pixel(64, 48, image::Rgba([200, 210, 220, 255]));
        let header = SheetHeader {
            title: String::new(),
            subtitle: String::new(),
        };
        let sheet = compose_sheet(&[left.clone(), right.clone()], &header).expect("sheet");
        let metrics = sheet_metrics(64);
        assert_eq!(sheet.width(), 2 * 64 + metrics.gap);
        assert_eq!(sheet.height(), 48 + metrics.header_height);
        // Every panel pixel survives where the layout says it is.
        for (x, y) in [(0u32, 0u32), (63, 47), (31, 20)] {
            assert_eq!(sheet.get_pixel(x, y + metrics.header_height), left.get_pixel(x, y));
            assert_eq!(
                sheet.get_pixel(x + 64 + metrics.gap, y + metrics.header_height),
                right.get_pixel(x, y)
            );
        }
        // The band and the gutter take the panels' own background.
        assert_eq!(sheet.get_pixel(0, 0).0, [10, 20, 30, 255]);
        assert_eq!(sheet.get_pixel(64, metrics.header_height).0, [10, 20, 30, 255]);

        let small = RgbaImage::from_pixel(32, 48, image::Rgba([0, 0, 0, 255]));
        let error = compose_sheet(&[left, small], &header).expect_err("two sizes");
        assert!(error.contains("resampled"), "{error}");
    }

    #[test]
    fn multiple_reference_panels_keep_every_input_pixel_and_requested_order() {
        let panels: Vec<_> = [10u8, 80, 150, 220].into_iter()
            .map(|value| RgbaImage::from_pixel(64, 48, image::Rgba([value, 20, 30, 255])))
            .collect();
        let sheet = compose_sheet(&panels, &SheetHeader { title: String::new(), subtitle: String::new() })
            .expect("run and three references");
        let layout = sheet_layout(4, 64, 48, Color::WHITE).expect("layout");
        for (index, panel) in panels.iter().enumerate() {
            let (left, top) = layout.panel_origin(index).expect("origin");
            for y in 0..panel.height() {
                for x in 0..panel.width() {
                    assert_eq!(sheet.get_pixel(left + x, top + y), panel.get_pixel(x, y));
                }
            }
        }
    }
}
