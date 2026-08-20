//! Horizontal interpolation from a first-guess grid to mesh points.
//!
//! A port of `mpas_init_atm_hinterp.F` restricted to the five methods case 7
//! actually names (`sixteen_pt`, `four_pt`, `wt_four_pt_average`,
//! `nearest_neighbor`, `search_extrap`) plus the cylindrical-equidistant
//! `latlon_to_ij` out of `mpas_init_atm_llxy.F`.
//!
//! ## Why it is a sequence and not a method
//! Each field carries an ordered list of methods.  A method that lands on a
//! missing value, or on a point the mask excludes, does not fall back to a
//! default: it *delegates to the next method in the list*, and the list ends
//! with the missing value itself.  The land-sea mask enters as an
//! interpolation mask, so a coastal cell can walk the whole sequence and end
//! at `search_extrap`, which breadth-first searches for the nearest unmasked,
//! non-missing source point.  That is why coastal cells get their own row in
//! the delta tables: a mismatch here is scattered, not global.
//!
//! ## Arithmetic width
//! Everything here is `f32`.  The reference `init_atmosphere_model` this lane
//! compares against is a single-precision build (`RKIND = 4`; the native inits
//! carry `float32` throughout), and the equality tests against `msgval` and
//! the `1.E-20` substitution inside `oned` are exact-comparison branches whose
//! outcome changes with the arithmetic width.  Widening to `f64` here would
//! take different branches, not merely round differently.

#![allow(clippy::needless_range_loop)]

use std::sync::atomic::{AtomicU64, Ordering};

/// The nine parabolic/bilinear/search methods, in the Fortran's own numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    SixteenPoint,
    FourPoint,
    NearestNeighbor,
    WtAverage4,
    Search,
}

/// A source slab with the Fortran's own index bounds, including the three
/// wrap columns on each side that make a global grid periodic in x.
#[derive(Debug, Clone)]
pub struct Slab {
    pub start_x: i32,
    pub end_x: i32,
    pub start_y: i32,
    pub end_y: i32,
    data: Vec<f32>,
}

impl Slab {
    pub fn new(start_x: i32, end_x: i32, start_y: i32, end_y: i32) -> Self {
        let nx = (end_x - start_x + 1) as usize;
        let ny = (end_y - start_y + 1) as usize;
        Slab {
            start_x,
            end_x,
            start_y,
            end_y,
            data: vec![0.0; nx * ny],
        }
    }

    #[inline]
    fn index(&self, i: i32, j: i32) -> usize {
        let nx = (self.end_x - self.start_x + 1) as usize;
        ((j - self.start_y) as usize) * nx + ((i - self.start_x) as usize)
    }

    #[inline]
    pub fn get(&self, i: i32, j: i32) -> f32 {
        self.data[self.index(i, j)]
    }

    #[inline]
    pub fn set(&mut self, i: i32, j: i32, v: f32) {
        let k = self.index(i, j);
        self.data[k] = v;
    }

    /// Build the `-2 .. nx+3` periodic halo the case code wraps every global
    /// slab into before interpolating.
    pub fn from_met(values: &[f32], nx: usize, ny: usize) -> Slab {
        let mut slab = Slab::new(-2, nx as i32 + 3, 1, ny as i32);
        for j in 1..=ny as i32 {
            for i in 1..=nx as i32 {
                slab.set(i, j, values[(j as usize - 1) * nx + (i as usize - 1)]);
            }
        }
        for j in 1..=ny as i32 {
            slab.set(0, j, slab.get(nx as i32, j));
            slab.set(-1, j, slab.get(nx as i32 - 1, j));
            slab.set(-2, j, slab.get(nx as i32 - 2, j));
            slab.set(nx as i32 + 1, j, slab.get(1, j));
            slab.set(nx as i32 + 2, j, slab.get(2, j));
            slab.set(nx as i32 + 3, j, slab.get(3, j));
        }
        slab
    }
}

/// Cylindrical equidistant grid geometry, as `map_set(PROJ_LATLON, ...)`
/// receives it from an intermediate header.
#[derive(Debug, Clone, Copy)]
pub struct LatLonProjection {
    pub lat1: f32,
    pub lon1: f32,
    pub latinc: f32,
    pub loninc: f32,
    pub knowni: f32,
    pub knownj: f32,
    pub nx: usize,
    pub ny: usize,
}

impl LatLonProjection {
    #[inline]
    fn ij(&self, lat: f32, lon: f32) -> (f32, f32) {
        let i = (lon - self.lon1) / self.loninc + self.knowni;
        let j = (lat - self.lat1) / self.latinc + self.knownj;
        (i, j)
    }

    /// `latlon_to_ij` plus the case code's own wrap-and-clamp, which is part
    /// of the field values and not a nicety: a cell just west of the first
    /// column is re-projected a whole turn east rather than falling off the
    /// grid, and a cell poleward of the last row is pinned to it.
    pub fn locate(&self, lat_deg: f32, lon_deg: f32) -> (f32, f32) {
        let (mut x, mut y) = self.ij(lat_deg, lon_deg);
        if x < 0.5 {
            let (x2, y2) = self.ij(lat_deg, lon_deg + 360.0);
            x = x2;
            y = y2;
        } else if x >= self.nx as f32 + 0.5 {
            let (x2, y2) = self.ij(lat_deg, lon_deg - 360.0);
            x = x2;
            y = y2;
        }
        if y < 0.5 {
            y = 1.0;
        } else if y >= self.ny as f32 + 0.5 {
            y = self.ny as f32;
        }
        (x, y)
    }
}

/// What happens when `oned`'s multiplicative guards underflow.
///
/// `sixteen_pt` substitutes `1.0e-20` for every source value that is exactly
/// zero, so that a zero-valued neighbour is treated as a small number rather
/// than as an absent one.  `oned` then guards its parabolas with `b*c != 0`
/// and `a*d == 0`.  When *two* substituted values meet in one of those
/// products the result is `1e-40`, which is subnormal in f32.
///
/// The reference `init_atmosphere_model` is built with `mpiifx -O3`, and
/// Intel Fortran turns on `-ftz` at every optimisation level above `-O0`.  In
/// that build `1e-20 * 1e-20` is flushed to zero, the guard reports "zero",
/// and `oned` returns its initialised `0.0` — which makes the *outer* `oned`
/// take the one-sided parabola instead of the two-sided one.  The
/// substitution is annihilated precisely where it was put in to help.
///
/// The same source compiled `ifx -O0`, `ifx -O3 -no-ftz` or `gfortran -O3`
/// keeps the subnormal and takes the two-sided branch.  Three of four builds
/// of identical source agree with [`Underflow::Preserve`]; the fourth differs
/// because of a floating-point mode, not because of the algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Underflow {
    /// Keep the subnormal, which is what the source says and what every build
    /// that is not `ifx -O1` or above produces.
    Preserve,
    /// Flush the guard products to zero, reproducing the reference build's
    /// `-ftz`.  Offered so the parity claim against the native init can be
    /// audited rather than asserted.
    ReproduceIfxFtz,
}

impl Underflow {
    pub fn parse(text: &str) -> Result<Underflow, String> {
        match text {
            "preserve" => Ok(Underflow::Preserve),
            "reproduce-ifx-ftz" => Ok(Underflow::ReproduceIfxFtz),
            other => Err(format!(
                "unknown underflow mode \"{other}\"; --oned-underflow takes \
                 preserve or reproduce-ifx-ftz"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Underflow::Preserve => "preserve",
            Underflow::ReproduceIfxFtz => "reproduce-ifx-ftz",
        }
    }
}

/// Everything the sequence needs that is not the point being interpolated to.
pub struct InterpContext<'a> {
    pub array: &'a Slab,
    pub msgval: f32,
    /// `Some(v)` when a mask slab applies; the Fortran signals "no mask" by
    /// passing `maskval = -1.0` and never reaching the masked branch.
    pub maskval: Option<f32>,
    pub mask: Option<&'a Slab>,
    pub underflow: Underflow,
    /// Counts the `oned` calls whose guard products landed in the subnormal
    /// range, i.e. every point at which the two modes can disagree.  Counted
    /// in **both** modes, so the receipt reports the size of the divergence
    /// whichever way the switch is set.
    pub underflow_hits: Option<&'a AtomicU64>,
}

impl InterpContext<'_> {
    #[inline]
    fn masked_at(&self, i: i32, j: i32) -> bool {
        match (self.mask, self.maskval) {
            (Some(m), Some(v)) => m.get(i, j) == v,
            _ => false,
        }
    }

    #[inline]
    fn has_mask(&self) -> bool {
        self.mask.is_some() && self.maskval.is_some()
    }
}

/// Run the method list at `(xx, yy)`, starting from `idx`.
pub fn interp_sequence(
    xx: f32,
    yy: f32,
    ctx: &InterpContext<'_>,
    list: &[Method],
    idx: usize,
) -> f32 {
    let Some(method) = list.get(idx) else {
        return ctx.msgval;
    };
    match method {
        Method::FourPoint => four_pt(xx, yy, ctx, list, idx + 1),
        Method::SixteenPoint => sixteen_pt(xx, yy, ctx, list, idx + 1),
        Method::WtAverage4 => wt_four_pt_average(xx, yy, ctx, list, idx + 1),
        Method::NearestNeighbor => nearest_neighbor(xx, yy, ctx, list, idx + 1),
        Method::Search => search_extrap(xx, yy, ctx, list, idx + 1),
    }
}

fn four_pt(xx: f32, yy: f32, ctx: &InterpContext<'_>, list: &[Method], idx: usize) -> f32 {
    let a = ctx.array;
    let min_x = xx.floor() as i32;
    let min_y = yy.floor() as i32;
    let max_x = xx.ceil() as i32;
    let max_y = yy.ceil() as i32;

    if min_x < a.start_x || max_x > a.end_x || min_y < a.start_y || max_y > a.end_y {
        return interp_sequence(xx, yy, ctx, list, idx);
    }

    let corner = [
        (min_x, min_y),
        (max_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
    ];
    let bad = if ctx.has_mask() {
        corner
            .iter()
            .any(|&(i, j)| a.get(i, j) == ctx.msgval || ctx.masked_at(i, j))
    } else {
        corner.iter().any(|&(i, j)| a.get(i, j) == ctx.msgval)
    };
    if bad {
        return interp_sequence(xx, yy, ctx, list, idx);
    }

    if min_x == max_x {
        if min_y == max_y {
            a.get(min_x, min_y)
        } else {
            a.get(min_x, min_y) * (max_y as f32 - yy) + a.get(min_x, max_y) * (yy - min_y as f32)
        }
    } else if min_y == max_y {
        a.get(min_x, min_y) * (max_x as f32 - xx) + a.get(max_x, min_y) * (xx - min_x as f32)
    } else {
        (yy - min_y as f32)
            * (a.get(min_x, max_y) * (max_x as f32 - xx)
                + a.get(max_x, max_y) * (xx - min_x as f32))
            + (max_y as f32 - yy)
                * (a.get(min_x, min_y) * (max_x as f32 - xx)
                    + a.get(max_x, min_y) * (xx - min_x as f32))
    }
}

/// One-dimensional overlapping parabolic interpolation.
///
/// The zero tests are the Fortran's, exactly: a zero neighbour degrades the
/// parabola to a line on that side, and a zero *centre* pair leaves the result
/// at whatever the leading `x == 0` / `x == 1` branch set — which is 0.0 when
/// `x` is strictly interior.  That is why the caller substitutes `1.E-20` for
/// exact zeros before calling and maps it back afterwards.
fn oned(x: f32, a: f32, b: f32, c: f32, d: f32, ctx: &InterpContext<'_>) -> f32 {
    let mut out = 0.0f32;
    if x == 0.0 {
        out = b;
    } else if x == 1.0 {
        out = c;
    }
    let mut bc = b * c;
    let mut ad = a * d;
    let subnormal = |v: f32| v != 0.0 && v.abs() < f32::MIN_POSITIVE;
    if subnormal(bc) || subnormal(ad) {
        if let Some(hits) = ctx.underflow_hits {
            hits.fetch_add(1, Ordering::Relaxed);
        }
        if ctx.underflow == Underflow::ReproduceIfxFtz {
            if subnormal(bc) {
                bc = 0.0;
            }
            if subnormal(ad) {
                ad = 0.0;
            }
        }
    }
    if bc != 0.0 {
        if ad == 0.0 {
            if a == 0.0 && d == 0.0 {
                out = b * (1.0 - x) + c * x;
            } else if a != 0.0 {
                out = b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b));
            } else if d != 0.0 {
                out = c + (1.0 - x) * (0.5 * (b - d) + (1.0 - x) * (0.5 * (b + d) - c));
            }
        } else {
            out = (1.0 - x) * (b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b)))
                + x * (c + (1.0 - x) * (0.5 * (b - d) + (1.0 - x) * (0.5 * (b + d) - c)));
        }
    }
    out
}

fn sixteen_pt(xx: f32, yy: f32, ctx: &InterpContext<'_>, list: &[Method], idx: usize) -> f32 {
    let a = ctx.array;
    if (xx as i32) < a.start_x
        || (xx as i32) > a.end_x
        || (yy as i32) < a.start_y
        || (yy as i32) > a.end_y
    {
        return interp_sequence(xx, yy, ctx, list, idx);
    }

    let i = (xx + 0.00001) as i32;
    let j = (yy + 0.00001) as i32;
    let x = xx - i as f32;
    let y = yy - j as f32;

    if x.abs() > 0.0001 || y.abs() > 0.0001 {
        let mut stl = [[0.0f32; 4]; 4];
        let mut n = 0usize;
        let mut is_masked = false;
        for k in 0..4usize {
            let mut kk = i + k as i32 - 1;
            if kk < a.start_x {
                kk = a.start_x;
            } else if kk > a.end_x {
                kk = a.end_x;
            }
            for l in 0..4usize {
                let mut ll = j + l as i32 - 1;
                if ll < a.start_y {
                    ll = a.start_y;
                } else if ll > a.end_y {
                    ll = a.end_y;
                }
                stl[k][l] = a.get(kk, ll);
                n += 1;
                if ctx.masked_at(kk, ll) {
                    is_masked = true;
                }
                if stl[k][l] == 0.0 && ctx.msgval != 0.0 {
                    stl[k][l] = 1.0e-20;
                }
            }
        }

        let bail = if ctx.has_mask() {
            is_masked || stl.iter().any(|row| row.contains(&ctx.msgval))
        } else {
            stl.iter().any(|row| row.contains(&ctx.msgval))
        };
        if bail {
            return interp_sequence(xx, yy, ctx, list, idx);
        }

        let a1 = oned(x, stl[0][0], stl[1][0], stl[2][0], stl[3][0], ctx);
        let b1 = oned(x, stl[0][1], stl[1][1], stl[2][1], stl[3][1], ctx);
        let c1 = oned(x, stl[0][2], stl[1][2], stl[2][2], stl[3][2], ctx);
        let d1 = oned(x, stl[0][3], stl[1][3], stl[2][3], stl[3][3], ctx);
        let mut out = oned(y, a1, b1, c1, d1, ctx);

        if n != 16 {
            let e = oned(y, stl[0][0], stl[0][1], stl[0][2], stl[0][3], ctx);
            let f = oned(y, stl[1][0], stl[1][1], stl[1][2], stl[1][3], ctx);
            let g = oned(y, stl[2][0], stl[2][1], stl[2][2], stl[2][3], ctx);
            let h = oned(y, stl[3][0], stl[3][1], stl[3][2], stl[3][3], ctx);
            out = (out + oned(x, e, f, g, h, ctx)) * 0.5;
        }
        if out == 1.0e-20 {
            out = 0.0;
        }
        out
    } else {
        let inside = i >= a.start_x && i <= a.end_x && j >= a.start_y && j <= a.end_y;
        let ok = if ctx.has_mask() {
            inside && !ctx.masked_at(i, j) && a.get(i, j) != ctx.msgval
        } else {
            inside && a.get(i, j) != ctx.msgval
        };
        if ok {
            a.get(i, j)
        } else {
            interp_sequence(xx, yy, ctx, list, idx)
        }
    }
}

fn wt_four_pt_average(
    xx: f32,
    yy: f32,
    ctx: &InterpContext<'_>,
    list: &[Method],
    idx: usize,
) -> f32 {
    let a = ctx.array;
    let mut ifx = xx.floor() as i32;
    let mut icx = xx.ceil() as i32;
    let mut ify = yy.floor() as i32;
    let mut icy = yy.ceil() as i32;

    let dist = |px: i32, py: i32| -> f32 {
        (1.0 - ((xx - px as f32).powi(2) + (yy - py as f32).powi(2)).sqrt()).max(0.0)
    };
    let mut fxfy = dist(ifx, ify);
    let mut fxcy = dist(ifx, icy);
    let mut cxfy = dist(icx, ify);
    let mut cxcy = dist(icx, icy);

    if ifx < a.start_x || icx > a.end_x || ify < a.start_y || icy > a.end_y {
        if xx > a.start_x as f32 - 0.5 && ifx < a.start_x {
            ifx = a.start_x;
            icx = a.start_x;
        } else if xx < a.end_x as f32 + 0.5 && icx > a.end_x {
            ifx = a.end_x;
            icx = a.end_x;
        }
        if yy > a.start_y as f32 - 0.5 && ify < a.start_y {
            ify = a.start_y;
            icy = a.start_y;
        } else if yy < a.end_y as f32 + 0.5 && icy > a.end_y {
            ify = a.end_y;
            icy = a.end_y;
        }
        if ifx < a.start_x || icx > a.end_x || ify < a.start_y || icy > a.end_y {
            return ctx.msgval;
        }
    }

    let kill = |i: i32, j: i32, w: &mut f32| {
        if a.get(i, j) == ctx.msgval || ctx.masked_at(i, j) {
            *w = 0.0;
        }
    };
    kill(ifx, ify, &mut fxfy);
    kill(ifx, icy, &mut fxcy);
    kill(icx, ify, &mut cxfy);
    kill(icx, icy, &mut cxcy);

    if fxfy == 0.0 && fxcy == 0.0 && cxfy == 0.0 && cxcy == 0.0 {
        interp_sequence(xx, yy, ctx, list, idx)
    } else {
        (fxfy * a.get(ifx, ify)
            + fxcy * a.get(ifx, icy)
            + cxfy * a.get(icx, ify)
            + cxcy * a.get(icx, icy))
            / (fxfy + fxcy + cxfy + cxcy)
    }
}

fn nearest_neighbor(
    xx: f32,
    yy: f32,
    ctx: &InterpContext<'_>,
    list: &[Method],
    idx: usize,
) -> f32 {
    let a = ctx.array;
    let ix = fortran_nint(xx);
    let iy = fortran_nint(yy);
    if ix < a.start_x || ix > a.end_x || iy < a.start_y || iy > a.end_y {
        return ctx.msgval;
    }
    let value = if ctx.has_mask() && ctx.masked_at(ix, iy) {
        ctx.msgval
    } else {
        a.get(ix, iy)
    };
    if value == ctx.msgval {
        interp_sequence(xx, yy, ctx, list, idx)
    } else {
        value
    }
}

/// Fortran `nint`: round half away from zero, not Rust's banker-free `round`
/// (which agrees) — spelled out because the two differ on some platforms'
/// `f32::round_ties_even`.
#[inline]
fn fortran_nint(v: f32) -> i32 {
    if v >= 0.0 {
        (v + 0.5).floor() as i32
    } else {
        (v - 0.5).ceil() as i32
    }
}

/// Breadth-first search outward for the nearest valid, unmasked source point.
///
/// The Fortran enqueues the four neighbours of every dequeued cell, marks them
/// in a bit array so each is visited once, stops at the *first* valid cell, and
/// then drains whatever is still queued looking for a strictly closer one.
/// That drain is not decoration: the queue is a plain FIFO over a 4-connected
/// lattice, so the first hit is nearest in Manhattan distance while the answer
/// is nearest in Euclidean distance, and the two differ near a diagonal.
fn search_extrap(xx: f32, yy: f32, ctx: &InterpContext<'_>, _list: &[Method], _idx: usize) -> f32 {
    let a = ctx.array;
    let sx = fortran_nint(xx);
    let sy = fortran_nint(yy);
    if sx < a.start_x || sx > a.end_x || sy < a.start_y || sy > a.end_y {
        return ctx.msgval;
    }

    let nx = (a.end_x - a.start_x + 1) as usize;
    let ny = (a.end_y - a.start_y + 1) as usize;
    let mut seen = vec![false; nx * ny];
    let mark = |seen: &mut Vec<bool>, i: i32, j: i32| -> bool {
        let k = ((j - a.start_y) as usize) * nx + ((i - a.start_x) as usize);
        if seen[k] {
            false
        } else {
            seen[k] = true;
            true
        }
    };

    let valid = |i: i32, j: i32| -> bool {
        if ctx.has_mask() {
            a.get(i, j) != ctx.msgval && !ctx.masked_at(i, j)
        } else {
            a.get(i, j) != ctx.msgval
        }
    };

    let mut queue: std::collections::VecDeque<(i32, i32)> = std::collections::VecDeque::new();
    queue.push_back((sx, sy));
    mark(&mut seen, sx, sy);

    let mut found: Option<(i32, i32)> = None;
    while let Some((i, j)) = queue.pop_front() {
        let hit = valid(i, j);
        if i > a.start_x && mark(&mut seen, i - 1, j) {
            queue.push_back((i - 1, j));
        }
        if i < a.end_x && mark(&mut seen, i + 1, j) {
            queue.push_back((i + 1, j));
        }
        if j > a.start_y && mark(&mut seen, i, j - 1) {
            queue.push_back((i, j - 1));
        }
        if j < a.end_y && mark(&mut seen, i, j + 1) {
            queue.push_back((i, j + 1));
        }
        if hit {
            found = Some((i, j));
            break;
        }
    }

    let Some((fi, fj)) = found else {
        return ctx.msgval;
    };
    let d2 = |i: i32, j: i32| (i as f32 - xx) * (i as f32 - xx) + (j as f32 - yy) * (j as f32 - yy);
    let mut distance = d2(fi, fj);
    let mut out = a.get(fi, fj);
    while let Some((i, j)) = queue.pop_front() {
        if valid(i, j) && d2(i, j) < distance {
            distance = d2(i, j);
            out = a.get(i, j);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(values: &[f32], nx: usize, ny: usize) -> Slab {
        let mut s = Slab::new(1, nx as i32, 1, ny as i32);
        for j in 1..=ny as i32 {
            for i in 1..=nx as i32 {
                s.set(i, j, values[(j as usize - 1) * nx + (i as usize - 1)]);
            }
        }
        s
    }

    #[test]
    fn a_global_slab_wraps_three_columns_each_way() {
        let s = Slab::from_met(&[1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
        assert_eq!(s.get(0, 1), 5.0);
        assert_eq!(s.get(-1, 1), 4.0);
        assert_eq!(s.get(-2, 1), 3.0);
        assert_eq!(s.get(6, 1), 1.0);
        assert_eq!(s.get(8, 1), 3.0);
    }

    #[test]
    fn four_point_is_bilinear_on_a_clean_square() {
        let a = plain(&[0.0, 10.0, 20.0, 30.0], 2, 2);
        let ctx = InterpContext {
            array: &a,
            msgval: -1.0e30,
            maskval: None,
            mask: None,
            underflow: Underflow::Preserve,
            underflow_hits: None,
        };
        let v = interp_sequence(1.5, 1.5, &ctx, &[Method::FourPoint], 0);
        assert!((v - 15.0).abs() < 1e-5, "{v}");
    }

    #[test]
    fn a_missing_corner_delegates_to_the_next_method() {
        // four_pt sees a missing corner and hands off; nearest_neighbor lands
        // on the valid cell at (2,2).
        let msg = -1.0e30f32;
        let a = plain(&[msg, 10.0, 20.0, 30.0], 2, 2);
        let ctx = InterpContext {
            array: &a,
            msgval: msg,
            maskval: None,
            mask: None,
            underflow: Underflow::Preserve,
            underflow_hits: None,
        };
        let v = interp_sequence(
            1.9,
            1.9,
            &ctx,
            &[Method::FourPoint, Method::NearestNeighbor],
            0,
        );
        assert_eq!(v, 30.0);
    }

    #[test]
    fn an_exhausted_sequence_returns_the_missing_value() {
        let msg = -1.0e30f32;
        let a = plain(&[msg, msg, msg, msg], 2, 2);
        let ctx = InterpContext {
            array: &a,
            msgval: msg,
            maskval: None,
            mask: None,
            underflow: Underflow::Preserve,
            underflow_hits: None,
        };
        let v = interp_sequence(1.5, 1.5, &ctx, &[Method::FourPoint], 0);
        assert_eq!(v, msg);
    }

    #[test]
    fn search_walks_past_masked_points_to_the_nearest_valid_one() {
        // A 5x1 strip whose only unmasked column is the far right.
        let a = plain(&[1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
        let mut mask = Slab::new(1, 5, 1, 1);
        for i in 1..=5 {
            mask.set(i, 1, if i == 5 { 1.0 } else { 0.0 });
        }
        let ctx = InterpContext {
            array: &a,
            msgval: -1.0e30,
            maskval: Some(0.0),
            mask: Some(&mask),
            underflow: Underflow::Preserve,
            underflow_hits: None,
        };
        let v = interp_sequence(1.0, 1.0, &ctx, &[Method::Search], 0);
        assert_eq!(v, 5.0);
    }

    #[test]
    fn the_latlon_inverse_wraps_a_point_west_of_the_first_column() {
        let proj = LatLonProjection {
            lat1: -90.0,
            lon1: 0.0,
            latinc: 1.0,
            loninc: 1.0,
            knowni: 1.0,
            knownj: 1.0,
            nx: 360,
            ny: 181,
        };
        // 0.75 W is x = -0.75 + 1 = 0.25 < 0.5, so it re-projects a turn east.
        let (x, _) = proj.locate(0.0, -0.75);
        assert!(x > 359.0, "{x}");
    }

    #[test]
    fn sixteen_point_on_an_exact_grid_point_returns_that_point() {
        let a = plain(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let ctx = InterpContext {
            array: &a,
            msgval: -1.0e30,
            maskval: None,
            mask: None,
            underflow: Underflow::Preserve,
            underflow_hits: None,
        };
        let v = interp_sequence(2.0, 2.0, &ctx, &[Method::SixteenPoint], 0);
        assert_eq!(v, 5.0);
    }

    /// The RH stencil at cell 122857, level 70000 Pa, of
    /// `MET:2026-08-12_06` — a moist/dry boundary where a whole row of the
    /// sixteen-point stencil is exactly zero.  The two expected values are
    /// not invented: they are what `ifx -O3` and `ifx -O3 -no-ftz` print for
    /// the reference `oned`/`sixteen_pt` fed this stencil.
    #[test]
    fn a_zero_row_makes_the_two_underflow_modes_disagree_by_seven_rh_points() {
        // rows are j = 498..501, columns i = 1083..1086; the interpolation
        // point sits at fx = 0.168579, fy = 0.786987 inside cell (1084, 499).
        let rows: [[f32; 6]; 6] = [
            [87.88595, 86.93902, 85.99889, 85.17765, 81.14071, 52.95525],
            [80.86084, 79.53122, 78.71996, 75.94976, 59.70227, 15.93333],
            [24.7772, 14.62796, 12.14132, 8.36185, 1.69258, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.09964],
            [0.59713, 0.0, 0.0, 0.09964, 0.59781, 1.5932],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let mut flat = Vec::new();
        for r in rows.iter() {
            flat.extend_from_slice(r);
        }
        // A 6x6 patch indexed 1..6; the sample point is inside cell (3, 2)
        // so the 4x4 stencil is columns 2..5 and rows 1..4.
        let mut a = Slab::new(1, 6, 1, 6);
        for j in 1..=6i32 {
            for i in 1..=6i32 {
                a.set(i, j, flat[(j as usize - 1) * 6 + (i as usize - 1)]);
            }
        }
        let xx = 3.0 + 0.168_579;
        let yy = 2.0 + 0.786_987;

        let hits = AtomicU64::new(0);
        let preserve = InterpContext {
            array: &a,
            msgval: -1.0e30,
            maskval: None,
            mask: None,
            underflow: Underflow::Preserve,
            underflow_hits: Some(&hits),
        };
        let kept = interp_sequence(xx, yy, &preserve, &[Method::SixteenPoint], 0);
        assert!(hits.load(Ordering::Relaxed) > 0, "the guard must underflow here");

        let flushed_hits = AtomicU64::new(0);
        let flushed_ctx = InterpContext {
            array: &a,
            msgval: -1.0e30,
            maskval: None,
            mask: None,
            underflow: Underflow::ReproduceIfxFtz,
            underflow_hits: Some(&flushed_hits),
        };
        let flushed = interp_sequence(xx, yy, &flushed_ctx, &[Method::SixteenPoint], 0);

        assert!(
            (kept - 23.282_33).abs() < 2.0e-4,
            "preserve must match ifx -O3 -no-ftz / gfortran -O3: got {kept}"
        );
        assert!(
            (flushed - 30.858_14).abs() < 2.0e-4,
            "reproduce-ifx-ftz must match the reference build: got {flushed}"
        );
        assert!(
            (kept - flushed).abs() > 7.0,
            "the modes must differ by the several RH points that made this a delta"
        );
        // The count is not mode-invariant in general: flushing changes what
        // the *outer* `oned` is handed, so a product that underflowed under
        // `Preserve` may not even be reached under `ReproduceIfxFtz`.  On this
        // stencil the two happen to agree, and both must be non-zero.
        assert!(flushed_hits.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn without_a_zero_row_the_two_underflow_modes_are_identical() {
        let a = plain(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
            5,
            5,
        );
        let h1 = AtomicU64::new(0);
        let h2 = AtomicU64::new(0);
        let keep = InterpContext {
            array: &a,
            msgval: -1.0e30,
            maskval: None,
            mask: None,
            underflow: Underflow::Preserve,
            underflow_hits: Some(&h1),
        };
        let flush = InterpContext {
            array: &a,
            msgval: -1.0e30,
            maskval: None,
            mask: None,
            underflow: Underflow::ReproduceIfxFtz,
            underflow_hits: Some(&h2),
        };
        let v1 = interp_sequence(2.4, 3.3, &keep, &[Method::SixteenPoint], 0);
        let v2 = interp_sequence(2.4, 3.3, &flush, &[Method::SixteenPoint], 0);
        assert_eq!(v1, v2);
        assert_eq!(h1.load(Ordering::Relaxed), 0);
        assert_eq!(h2.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn an_unknown_underflow_mode_is_refused_by_name() {
        let err = Underflow::parse("ftz").unwrap_err();
        assert!(err.contains("ftz"), "{err}");
        assert!(err.contains("reproduce-ifx-ftz"), "{err}");
    }
}
