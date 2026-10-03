/// Inverse geolocation: lat/lon bin → row-major grid index.
pub struct InverseLut {
    lat0: f32,
    lon0: f32,
    inv_dlat: f32,
    inv_dlon: f32,
    width: usize,
    height: usize,
    index: Vec<u32>,
}

/// Hard floor on the bin size — degenerate-data guard only (~11 m). The real
/// lower bound comes from [`MAX_LUT_BINS`]; a fixed coarse floor here is what
/// quantized 250 m WRF grids into ~3.3 km blocks (the old 0.03° floor made
/// every gate in a bucket resolve to ONE model cell, and the sampler's local
/// 3x3 stencil search could never reach the true cell from a seed ~7 cells
/// away — giant constant wedges on the synthetic radar).
const MIN_BIN_DEG: f32 = 0.0001;
/// Total-bin budget for the inverse index (u32 each, ~64 MB worst case).
/// Small fine grids (250 m WRF: ~0.7 M bins) index at native spacing; huge
/// domains (full-disk satellite) are budget-limited to roughly the same bin
/// size the old fixed 0.03° floor produced, so their memory profile is
/// unchanged.
const MAX_LUT_BINS: f32 = 16_000_000.0;
/// Per-axis ceiling for the inverse index. The allocation has always been
/// capped to this size, but the bin spacing must honor the same ceiling: if
/// `width.min(MAX_LUT_AXIS_BINS)` truncates a long, narrow domain without
/// increasing `bin`, every longitude beyond the retained western bins becomes
/// unaddressable. Reserve one bin below the ceiling to absorb f32 rounding at
/// the maximum coordinate.
const MAX_LUT_AXIS_BINS: usize = 8192;
const HOLE_FILL_PASSES: usize = 3;

impl InverseLut {
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.index.len().saturating_mul(std::mem::size_of::<u32>()))
    }

    /// Build from the grid's lat/lon arrays (~a second for CONUS HRRR;
    /// run on a background thread).
    pub fn build(lat: &[f32], lon: &[f32]) -> Option<Self> {
        Self::build_inner(lat, lon, None, false)
    }

    /// Build from a shaped lat/lon grid. Satellite grids can be strongly
    /// curvilinear, so estimate spacing from real row/column neighbors
    /// instead of only 1D consecutive samples.
    pub fn build_with_shape(lat: &[f32], lon: &[f32], nx: usize, ny: usize) -> Option<Self> {
        if nx == 0 || ny == 0 || nx.saturating_mul(ny) != lat.len() || lat.len() != lon.len() {
            return Self::build(lat, lon);
        }
        Self::build_inner(lat, lon, Some((nx, ny)), false)
    }

    /// Build from a shaped grid whose row/column perimeter IS the true data
    /// boundary (WRF and similar regional model domains). Identical to
    /// [`Self::build_with_shape`] except the hole-fill dilation is confined
    /// to bins inside that perimeter polygon, so a query point outside the
    /// curvilinear domain edge — but still inside the rectangular lat/lon
    /// bbox the bins cover — returns `None` instead of the nearest edge
    /// cell (the smeared ~3-bin ring on the synthetic radar's domain
    /// boundary). Seeded bins and interior dilation are untouched, so
    /// in-domain lookups resolve exactly as before.
    ///
    /// Satellite layers must keep [`Self::build_with_shape`]: a full-disk
    /// grid's valid data ends at the earth's limb (NaN regions INSIDE the
    /// grid), not at the grid perimeter. Misuse still degrades safely — any
    /// non-finite perimeter point disables the mask and this build is then
    /// bin-for-bin identical to `build_with_shape`.
    pub fn build_with_shape_domain_bounded(
        lat: &[f32],
        lon: &[f32],
        nx: usize,
        ny: usize,
    ) -> Option<Self> {
        if nx == 0 || ny == 0 || nx.saturating_mul(ny) != lat.len() || lat.len() != lon.len() {
            return Self::build(lat, lon);
        }
        Self::build_inner(lat, lon, Some((nx, ny)), true)
    }

    fn build_inner(
        lat: &[f32],
        lon: &[f32],
        shape: Option<(usize, usize)>,
        bound_to_perimeter: bool,
    ) -> Option<Self> {
        let mut lat_min = f32::INFINITY;
        let mut lat_max = f32::NEG_INFINITY;
        let mut lon_min = f32::INFINITY;
        let mut lon_max = f32::NEG_INFINITY;
        for (&la, &lo) in lat.iter().zip(lon.iter()) {
            if la.is_finite() && lo.is_finite() {
                lat_min = lat_min.min(la);
                lat_max = lat_max.max(la);
                lon_min = lon_min.min(lo);
                lon_max = lon_max.max(lo);
            }
        }
        if !lat_min.is_finite() || lat_max <= lat_min || lon_max <= lon_min {
            return None;
        }
        // Bin size adapts to the grid's spacing (HRRR ~0.03°, GFS 0.25°):
        // bins comparable to the spacing keep holes within one cell of a
        // sample, which the fill passes close. Median of non-degenerate
        // consecutive steps — row-wrap jumps (tens of degrees between the
        // end of one row and the start of the next) are filtered out.
        let spacing = shape
            .and_then(|(nx, ny)| shaped_grid_spacing(lat, lon, nx, ny))
            .unwrap_or_else(|| consecutive_spacing(lat, lon).unwrap_or(MIN_BIN_DEG));
        // Bin = the grid's own spacing, floored by the total-bin budget so a
        // huge domain cannot allocate an unbounded index. The budget floor —
        // not a fixed degree floor — is what lets a 250 m grid index at native
        // resolution while a full-disk satellite grid stays ~64 MB.
        let lat_span = lat_max - lat_min;
        let lon_span = lon_max - lon_min;
        let budget_floor = ((lat_span * lon_span) / MAX_LUT_BINS).sqrt();
        let axis_floor = lat_span.max(lon_span) / (MAX_LUT_AXIS_BINS - 2) as f32;
        let bin = (spacing * if shape.is_some() { 1.25 } else { 1.1 })
            .max(budget_floor)
            .max(axis_floor)
            .max(MIN_BIN_DEG);
        let width = ((lon_span / bin).ceil() as usize + 1).min(MAX_LUT_AXIS_BINS);
        let height = ((lat_span / bin).ceil() as usize + 1).min(MAX_LUT_AXIS_BINS);
        let mut index = vec![u32::MAX; width * height];
        for (i, (&la, &lo)) in lat.iter().zip(lon.iter()).enumerate() {
            if !la.is_finite() || !lo.is_finite() {
                continue;
            }
            let bx = ((lo - lon_min) / bin) as usize;
            let by = ((la - lat_min) / bin) as usize;
            if bx < width && by < height {
                index[by * width + bx] = i as u32;
            }
        }
        // True-domain mask (opt-in, WRF/model path): confine the dilation
        // below to bins inside the grid's perimeter polygon, so bins between
        // the curvilinear domain edge and the rectangular bbox stay empty
        // (lookup → None) instead of inheriting the nearest edge cell.
        let domain_mask = if bound_to_perimeter {
            shape.and_then(|(nx, ny)| {
                perimeter_bin_mask(lat, lon, nx, ny, lat_min, lon_min, bin, width, height)
            })
        } else {
            None
        };
        // Hole fill: model grid spacing can exceed the bin size away from
        // the grid center; dilate a few passes so bins between grid points
        // resolve to a neighbor.
        for _ in 0..HOLE_FILL_PASSES {
            let snapshot = index.clone();
            for by in 0..height {
                for bx in 0..width {
                    if snapshot[by * width + bx] != u32::MAX {
                        continue;
                    }
                    if let Some(mask) = domain_mask.as_deref()
                        && !mask[by * width + bx]
                    {
                        continue;
                    }
                    let mut fill = u32::MAX;
                    for (dy, dx) in [(0i64, 1i64), (0, -1), (1, 0), (-1, 0)] {
                        let ny = by as i64 + dy;
                        let nx = bx as i64 + dx;
                        if ny < 0 || nx < 0 || ny >= height as i64 || nx >= width as i64 {
                            continue;
                        }
                        let v = snapshot[ny as usize * width + nx as usize];
                        if v != u32::MAX {
                            fill = v;
                            break;
                        }
                    }
                    if fill != u32::MAX {
                        index[by * width + bx] = fill;
                    }
                }
            }
        }
        Some(Self {
            lat0: lat_min,
            lon0: lon_min,
            inv_dlat: 1.0 / bin,
            inv_dlon: 1.0 / bin,
            width,
            height,
            index,
        })
    }

    /// Grid index for a lat/lon, or None outside the grid.
    #[inline]
    pub fn lookup(&self, lat: f32, lon: f32) -> Option<usize> {
        let bx = ((lon - self.lon0) * self.inv_dlon) as isize;
        let by = ((lat - self.lat0) * self.inv_dlat) as isize;
        if bx < 0 || by < 0 || bx as usize >= self.width || by as usize >= self.height {
            return None;
        }
        let v = self.index[by as usize * self.width + bx as usize];
        (v != u32::MAX).then_some(v as usize)
    }
}

fn consecutive_spacing(lat: &[f32], lon: &[f32]) -> Option<f32> {
    let mut steps: Vec<f32> = Vec::with_capacity(4096);
    for source in [lat, lon] {
        for pair in source.windows(2).take(4096) {
            if pair[0].is_finite() && pair[1].is_finite() {
                let step = (pair[1] - pair[0]).abs();
                if step > 1e-6 && step < 2.0 {
                    steps.push(step);
                }
            }
        }
    }
    percentile_step(steps, 0.5)
}

fn shaped_grid_spacing(lat: &[f32], lon: &[f32], nx: usize, ny: usize) -> Option<f32> {
    let mut steps: Vec<f32> = Vec::with_capacity(8192);
    let step_x = (nx / 96).max(1);
    let step_y = (ny / 96).max(1);
    for y in (0..ny).step_by(step_y) {
        for x in (0..nx.saturating_sub(1)).step_by(step_x) {
            push_neighbor_step(&mut steps, lat, lon, y * nx + x, y * nx + x + 1);
        }
    }
    for y in (0..ny.saturating_sub(1)).step_by(step_y) {
        for x in (0..nx).step_by(step_x) {
            push_neighbor_step(&mut steps, lat, lon, y * nx + x, (y + 1) * nx + x);
        }
    }
    percentile_step(steps, 0.75)
}

fn push_neighbor_step(steps: &mut Vec<f32>, lat: &[f32], lon: &[f32], a: usize, b: usize) {
    let (lat_a, lon_a, lat_b, lon_b) = (lat[a], lon[a], lat[b], lon[b]);
    if !lat_a.is_finite() || !lon_a.is_finite() || !lat_b.is_finite() || !lon_b.is_finite() {
        return;
    }
    let step = (lat_a - lat_b).abs().max(wrapped_lon_delta(lon_a, lon_b));
    if step > 1e-5 && step < 5.0 {
        steps.push(step);
    }
}

fn wrapped_lon_delta(a: f32, b: f32) -> f32 {
    let raw = (a - b).abs().rem_euclid(360.0);
    raw.min(360.0 - raw)
}

fn percentile_step(mut steps: Vec<f32>, percentile: f32) -> Option<f32> {
    if steps.is_empty() {
        return None;
    }
    steps.sort_by(f32::total_cmp);
    let index = ((steps.len() - 1) as f32 * percentile.clamp(0.0, 1.0)).round() as usize;
    steps.get(index).copied()
}

/// The grid's outer boundary as a closed (lon, lat) ring — south row W→E,
/// east column S→N, north row E→W, west column N→S in grid order. `None`
/// when the grid is degenerate or ANY perimeter point is non-finite (a
/// full-disk satellite grid, whose valid data ends at the earth's limb, not
/// at the grid perimeter) — callers then skip domain masking entirely.
fn perimeter_ring(lat: &[f32], lon: &[f32], nx: usize, ny: usize) -> Option<Vec<(f64, f64)>> {
    if nx < 2 || ny < 2 {
        return None;
    }
    let mut ids: Vec<usize> = Vec::with_capacity(2 * (nx + ny));
    ids.extend(0..nx);
    ids.extend((1..ny).map(|y| y * nx + (nx - 1)));
    ids.extend((0..nx - 1).rev().map(|x| (ny - 1) * nx + x));
    ids.extend((1..ny - 1).rev().map(|y| y * nx));
    let mut ring = Vec::with_capacity(ids.len() + 1);
    for id in ids {
        let (la, lo) = (lat[id], lon[id]);
        if !la.is_finite() || !lo.is_finite() {
            return None;
        }
        ring.push((f64::from(lo), f64::from(la)));
    }
    let first = ring[0];
    ring.push(first);
    Some(ring)
}

/// Rasterize the grid's perimeter polygon into the bin grid: `true` for
/// bins whose CENTER lies inside the polygon (even-odd scanline fill) plus
/// every bin the boundary itself passes through (edges sampled at half-bin
/// steps), so bins straddling the boundary stay fillable and the in-domain
/// side never shrinks. O(bin rows × perimeter edges) — a few ms for the
/// largest WRF domains. `None` disables masking (see [`perimeter_ring`]).
#[allow(clippy::too_many_arguments)]
fn perimeter_bin_mask(
    lat: &[f32],
    lon: &[f32],
    nx: usize,
    ny: usize,
    lat_min: f32,
    lon_min: f32,
    bin: f32,
    width: usize,
    height: usize,
) -> Option<Vec<bool>> {
    let ring = perimeter_ring(lat, lon, nx, ny)?;
    let (lat0, lon0, bin) = (f64::from(lat_min), f64::from(lon_min), f64::from(bin));
    if !bin.is_finite() || bin <= 0.0 || width == 0 || height == 0 {
        return None;
    }
    let mut mask = vec![false; width * height];

    // Even-odd scanline fill at bin-center latitudes. The strict `>` on
    // both edge endpoints keeps crossing counts even when a scanline grazes
    // a vertex.
    let mut crossings: Vec<f64> = Vec::new();
    for by in 0..height {
        let y = lat0 + (by as f64 + 0.5) * bin;
        crossings.clear();
        for edge in ring.windows(2) {
            let ((x0, y0), (x1, y1)) = (edge[0], edge[1]);
            if (y0 > y) != (y1 > y) {
                crossings.push(x0 + (y - y0) * (x1 - x0) / (y1 - y0));
            }
        }
        crossings.sort_by(f64::total_cmp);
        for pair in crossings.as_chunks::<2>().0 {
            let lo = ((pair[0] - lon0) / bin - 0.5).ceil().max(0.0) as usize;
            let hi = ((pair[1] - lon0) / bin - 0.5).floor();
            if hi < 0.0 || lo >= width {
                continue;
            }
            let hi = (hi as usize).min(width - 1);
            for bx in lo..=hi {
                mask[by * width + bx] = true;
            }
        }
    }

    // Boundary bins: walk each edge at half-bin steps so a bin the domain
    // edge merely clips still counts as in-domain.
    for edge in ring.windows(2) {
        let ((x0, y0), (x1, y1)) = (edge[0], edge[1]);
        let steps = (((x1 - x0).abs().max((y1 - y0).abs()) / bin * 2.0).ceil() as usize).max(1);
        for step in 0..=steps {
            let t = step as f64 / steps as f64;
            let bx = (x0 + t * (x1 - x0) - lon0) / bin;
            let by = (y0 + t * (y1 - y0) - lat0) / bin;
            if bx >= 0.0 && by >= 0.0 && (bx as usize) < width && (by as usize) < height {
                mask[by as usize * width + bx as usize] = true;
            }
        }
    }
    Some(mask)
}


pub(crate) fn neighboring_cell_starts(index: usize, len: usize) -> [Option<usize>; 2] {
    if len < 2 {
        return [None, None];
    }
    let first = index.saturating_sub(1).min(len - 2);
    let second = index.min(len - 2);
    if first == second {
        [Some(first), None]
    } else {
        [Some(first), Some(second)]
    }
}

pub(crate) fn solve_bilinear_coords(
    corners: [(f64, f64); 4],
    target_x: f64,
    target_y: f64,
) -> Option<(f64, f64)> {
    let [(x00, y00), (x10, y10), (x01, y01), (x11, y11)] = corners;
    let mut u = 0.5;
    let mut v = 0.5;
    for _ in 0..8 {
        let one_u = 1.0 - u;
        let one_v = 1.0 - v;
        let x = one_u * one_v * x00 + u * one_v * x10 + one_u * v * x01 + u * v * x11;
        let y = one_u * one_v * y00 + u * one_v * y10 + one_u * v * y01 + u * v * y11;
        let rx = target_x - x;
        let ry = target_y - y;
        if rx.abs().max(ry.abs()) < 1e-6 {
            return Some((u, v));
        }
        let dx_du = -one_v * x00 + one_v * x10 - v * x01 + v * x11;
        let dx_dv = -one_u * x00 - u * x10 + one_u * x01 + u * x11;
        let dy_du = -one_v * y00 + one_v * y10 - v * y01 + v * y11;
        let dy_dv = -one_u * y00 - u * y10 + one_u * y01 + u * y11;
        let det = dx_du * dy_dv - dx_dv * dy_du;
        if det.abs() < 1e-12 {
            return None;
        }
        let du = (rx * dy_dv - dx_dv * ry) / det;
        let dv = (dx_du * ry - rx * dy_du) / det;
        u += du;
        v += dv;
        if !u.is_finite() || !v.is_finite() || u.abs().max(v.abs()) > 3.0 {
            return None;
        }
    }
    Some((u, v))
}

pub(crate) fn unwrap_lon_near(mut lon: f64, target: f64) -> f64 {
    while lon - target > 180.0 {
        lon -= 360.0;
    }
    while lon - target < -180.0 {
        lon += 360.0;
    }
    lon
}
