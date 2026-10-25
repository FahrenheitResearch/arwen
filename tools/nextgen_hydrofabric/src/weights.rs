//! Polygon/rectangle intersections in the parent's projected metre coordinate system.
//! Independent implementation of standard WKB and half-plane polygon clipping.
use crate::{
    network::{id, Prepared},
    Result,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ffi::{c_char, c_int, c_void, CString};
#[derive(Clone, Serialize, Deserialize)]
pub struct Grid {
    pub nx: u32,
    pub ny: u32,
    pub x0: f64,
    pub y0: f64,
    pub dx: f64,
    pub dy: f64,
    pub crs: String,
    #[serde(default)]
    pub row_order: RowOrder,
}
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowOrder {
    #[default]
    SouthToNorth,
    NorthToSouth,
}
impl Grid {
    fn index(&self, x: u32, south_row: u32) -> u32 {
        let row = match self.row_order {
            RowOrder::SouthToNorth => south_row,
            RowOrder::NorthToSouth => self.ny - 1 - south_row,
        };
        row * self.nx + x
    }
}
#[derive(Serialize)]
pub struct Weights {
    pub offsets: Vec<u64>,
    pub indices: Vec<u32>,
    pub weights: Vec<f64>,
    pub overlap_m2: Vec<f64>,
    pub catchment_area_m2: Vec<f64>,
    pub coverage_fraction: Vec<f64>,
    pub grid: Grid,
    pub method: &'static str,
}
type Point = [f64; 2];
type Polygon = Vec<Vec<Point>>;
struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.p.checked_add(N).ok_or("WKB offset overflow")?;
        let v = self.b.get(self.p..end).ok_or("truncated WKB")?.try_into()?;
        self.p = end;
        Ok(v)
    }
    fn u32(&mut self, le: bool) -> Result<u32> {
        let b = self.take()?;
        Ok(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }
    fn f64(&mut self, le: bool) -> Result<f64> {
        let b = self.take()?;
        let v = if le {
            f64::from_le_bytes(b)
        } else {
            f64::from_be_bytes(b)
        };
        if !v.is_finite() {
            return Err("nonfinite polygon coordinate".into());
        }
        Ok(v)
    }
    fn geometry(&mut self, out: &mut Vec<Polygon>, depth: usize) -> Result<()> {
        if depth > 32 {
            return Err("WKB nesting exceeds 32".into());
        }
        let le = match self.take::<1>()?[0] {
            0 => false,
            1 => true,
            _ => return Err("invalid WKB byte order".into()),
        };
        let code = self.u32(le)?;
        let base = code % 1000;
        let dims = match code / 1000 {
            0 => 2,
            1 | 2 => 3,
            3 => 4,
            _ => return Err("unsupported EWKB (expected GeoPackage ISO WKB)".into()),
        };
        match base {
            3 => {
                let n = self.u32(le)? as usize;
                if n > self.b.len() / 4 {
                    return Err("invalid WKB ring count".into());
                }
                let mut rings = Vec::new();
                for _ in 0..n {
                    let m = self.u32(le)? as usize;
                    if m > self.b.len() / (8 * dims) {
                        return Err("invalid WKB point count".into());
                    }
                    let mut ring = Vec::with_capacity(m);
                    for _ in 0..m {
                        let xy = [self.f64(le)?, self.f64(le)?];
                        for _ in 2..dims {
                            self.f64(le)?;
                        }
                        ring.push(xy)
                    }
                    if ring.len() < 4 || ring.first() != ring.last() {
                        return Err("polygon ring must be closed with at least four points".into());
                    }
                    rings.push(ring)
                }
                out.push(rings);
            }
            6 | 7 => {
                let n = self.u32(le)? as usize;
                if n > self.b.len() / 5 {
                    return Err("invalid WKB collection count".into());
                }
                for _ in 0..n {
                    self.geometry(out, depth + 1)?
                }
            }
            _ => {
                return Err(
                    format!("divides require polygon/multipolygon, found WKB type {code}").into(),
                )
            }
        }
        Ok(())
    }
}
pub fn polygons(blob: &[u8]) -> Result<Vec<Polygon>> {
    if blob.len() < 8 || &blob[..2] != b"GP" || blob[2] != 0 {
        return Err("invalid GeoPackage geometry header".into());
    }
    if blob[3] & 0xe0 != 0 {
        return Err("extended/reserved GeoPackage geometry flags unsupported".into());
    }
    if blob[3] & 16 != 0 {
        return Ok(vec![]);
    }
    let envelope = match (blob[3] >> 1) & 7 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        _ => return Err("invalid GeoPackage envelope indicator".into()),
    };
    let mut c = Cursor {
        b: blob,
        p: 8 + envelope,
    };
    let mut out = Vec::new();
    c.geometry(&mut out, 0)?;
    if c.p != blob.len() {
        return Err("trailing geometry bytes".into());
    }
    Ok(out)
}
#[repr(C)]
#[derive(Copy, Clone)]
struct Coord {
    x: f64,
    y: f64,
    z: f64,
    t: f64,
}
// Pinned bundled PROJ/SQLite; no host library lookup.
extern crate hydro_proj;
extern "C" {
    fn proj_create(c: *mut c_void, definition: *const c_char) -> *mut c_void;
    fn proj_crs_get_coordinate_system(c: *mut c_void, crs: *const c_void) -> *mut c_void;
    fn proj_cs_get_type(c: *mut c_void, cs: *const c_void) -> c_int;
    fn proj_cs_get_axis_count(c: *mut c_void, cs: *const c_void) -> c_int;
    fn proj_cs_get_axis_info(
        c: *mut c_void,
        cs: *const c_void,
        index: c_int,
        name: *mut *const c_char,
        abbrev: *mut *const c_char,
        direction: *mut *const c_char,
        factor: *mut f64,
        unit_name: *mut *const c_char,
        unit_authority: *mut *const c_char,
        unit_code: *mut *const c_char,
    ) -> c_int;
    fn proj_context_create() -> *mut c_void;
    fn proj_context_destroy(c: *mut c_void);
    fn proj_context_set_enable_network(c: *mut c_void, e: c_int) -> c_int;
    fn proj_create_crs_to_crs(
        c: *mut c_void,
        s: *const c_char,
        t: *const c_char,
        a: *mut c_void,
    ) -> *mut c_void;
    fn proj_normalize_for_visualization(c: *mut c_void, p: *const c_void) -> *mut c_void;
    fn proj_trans(p: *mut c_void, d: c_int, c: Coord) -> Coord;
    fn proj_destroy(p: *mut c_void) -> *mut c_void;
}


/// Use the CRS coordinate system's actual units, never assume an EPSG identifier means metres.
pub(crate) fn validate_grid_crs(definition: &str) -> Result<()> {
    let text = CString::new(definition)?;
    unsafe {
        let context = proj_context_create();
        if context.is_null() {
            return Err("PROJ context allocation failed".into());
        }
        proj_context_set_enable_network(context, 0);
        let crs = proj_create(context, text.as_ptr());
        let cs = if crs.is_null() {
            std::ptr::null_mut()
        } else {
            proj_crs_get_coordinate_system(context, crs)
        };
        // PROJ's public PJ_CS_TYPE_CARTESIAN is enum value 1 (proj.h).
        let mut valid = !cs.is_null()
            && proj_cs_get_type(context, cs) == 1
            && proj_cs_get_axis_count(context, cs) == 2;
        if valid {
            for axis in 0..2 {
                let mut factor = 0.;
                let status = proj_cs_get_axis_info(
                    context,
                    cs,
                    axis,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut factor,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
                valid &= status != 0 && (factor - 1.).abs() < 1e-12;
            }
        }
        if !cs.is_null() {
            proj_destroy(cs);
        }
        if !crs.is_null() {
            proj_destroy(crs);
        }
        proj_context_destroy(context);
        if valid {
            Ok(())
        } else {
            Err("area-weight grid CRS must have two Cartesian axes in metres".into())
        }
    }
}
pub(crate) struct Projection {
    context: *mut c_void,
    p: *mut c_void,
}
impl Projection {
    pub(crate) fn new(source: &str, target: &str) -> Result<Self> {
        unsafe {
            let s = CString::new(source)?;
            let t = CString::new(target)?;
            let context = proj_context_create();
            if context.is_null() {
                return Err("PROJ context allocation failed".into());
            }
            proj_context_set_enable_network(context, 0);
            let raw = proj_create_crs_to_crs(context, s.as_ptr(), t.as_ptr(), std::ptr::null_mut());
            if raw.is_null() {
                proj_context_destroy(context);
                return Err(format!("cannot transform {source} to {target}").into());
            }
            let p = proj_normalize_for_visualization(context, raw);
            proj_destroy(raw);
            if p.is_null() {
                proj_context_destroy(context);
                return Err("PROJ axis normalization failed".into());
            }
            Ok(Self { context, p })
        }
    }
    pub(crate) fn transform(&self, point: Point) -> Result<Point> {
        let c = unsafe {
            proj_trans(
                self.p,
                1,
                Coord {
                    x: point[0],
                    y: point[1],
                    z: 0.,
                    t: 0.,
                },
            )
        };
        if !c.x.is_finite() || !c.y.is_finite() {
            return Err("PROJ coordinate outside transform domain".into());
        }
        Ok([c.x, c.y])
    }
}
impl Drop for Projection {
    fn drop(&mut self) {
        unsafe {
            proj_destroy(self.p);
            proj_context_destroy(self.context);
        }
    }
}
fn area(r: &[Point]) -> f64 {
    if r.len() < 3 {
        return 0.;
    }
    let origin = r[0];
    let mut s = 0.;
    for i in 1..r.len() - 1 {
        s += (r[i][0] - origin[0]) * (r[i + 1][1] - origin[1])
            - (r[i + 1][0] - origin[0]) * (r[i][1] - origin[1]);
    }
    s.abs() * 0.5
}
fn clip(r: &[Point], axis: usize, bound: f64, lower: bool) -> Vec<Point> {
    let mut out = Vec::new();
    if r.is_empty() {
        return out;
    }
    let mut a = r[r.len() - 1];
    let mut inside_a = if lower {
        a[axis] >= bound
    } else {
        a[axis] <= bound
    };
    for &b in r {
        let inside_b = if lower {
            b[axis] >= bound
        } else {
            b[axis] <= bound
        };
        if inside_a != inside_b {
            let f = (bound - a[axis]) / (b[axis] - a[axis]);
            let mut p = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
            p[axis] = bound;
            out.push(p)
        }
        if inside_b {
            out.push(b)
        }
        a = b;
        inside_a = inside_b;
    }
    out
}
fn clipped_area(r: &[Point], bounds: [f64; 4]) -> f64 {
    let r = clip(r, 0, bounds[0], true);
    let r = clip(&r, 0, bounds[2], false);
    let r = clip(&r, 1, bounds[1], true);
    area(&clip(&r, 1, bounds[3], false))
}
fn poly_area(p: &Polygon, bounds: Option<[f64; 4]>) -> f64 {
    p.iter()
        .enumerate()
        .map(|(i, r)| {
            let a = bounds.map_or_else(|| area(r), |b| clipped_area(r, b));
            if i == 0 {
                a
            } else {
                -a
            }
        })
        .sum::<f64>()
        .max(0.)
}
/// `f` over `items` on all threads, in order, with one PROJ transformation per
/// work chunk (None when the CRSs agree). Breakage it prevents: rayon's
/// map_init built a PROJ context and opened proj.db for every split job, so
/// area weights and the channel raster of a 740 k-catchment fabric ran at
/// about 1.3 cores (17.4 s and 15.8 s on a 24-core host; 0.9 s and 0.8 s now, every
/// output file byte-identical). Same per-item function, same order.
pub(crate) fn par_map_projected<T, R, F>(items: &[T], source: &str, target: &str, f: F) -> Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(Option<&Projection>, &T) -> Result<R> + Sync,
{
    let chunk = items.len().div_ceil(rayon::current_num_threads().max(1) * 8).max(1);
    let parts = items
        .par_chunks(chunk)
        .map(|part| {
            let projection = if source != target { Some(Projection::new(source, target)?) } else { None };
            part.iter().map(|item| f(projection.as_ref(), item)).collect::<Result<Vec<R>>>()
        })
        .collect::<Result<Vec<Vec<R>>>>()?;
    Ok(parts.into_iter().flatten().collect())
}

pub fn build(prepared: &Prepared, grid: &Grid) -> Result<Weights> {
    if grid.nx == 0
        || grid.ny == 0
        || grid.nx.checked_mul(grid.ny).is_none()
        || ![grid.x0, grid.y0, grid.dx, grid.dy]
            .iter()
            .all(|x| x.is_finite())
        || grid.dx <= 0.
        || grid.dy <= 0.
    {
        return Err("grid needs finite origin, positive metre spacing and <=u32::MAX cells".into());
    }
    validate_grid_crs(&grid.crs)?;
    let layer = &prepared.layers["divides"];
    let rows = layer
        .rows
        .iter()
        .map(|r| Ok((id(r, "divides")?, r)))
        .collect::<Result<std::collections::HashMap<_, _>>>()?;
    let source = layer
        .crs
        .as_deref()
        .ok_or("geometry CRS missing from gpkg_spatial_ref_sys")?;
    let pieces = par_map_projected(
        &prepared.network.catchment_ids,
        source,
        &grid.crs,
            |projection, s| -> Result<(Vec<(u32, f64)>, f64)> {
                let row = rows.get(s).ok_or("missing selected divide")?;
                let mut polys = polygons(row.geometry.as_deref().ok_or("divide geometry absent")?)?;
                if let Some(p) = projection {
                    for poly in &mut polys {
                        for ring in poly {
                            for point in ring {
                                *point = p.transform(*point)?
                            }
                        }
                    }
                }
                let total = polys.iter().map(|p| poly_area(p, None)).sum::<f64>();
                if total <= 0. {
                    return Err(format!("empty/zero area divide {s}").into());
                }
                let mut bbox = [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ];
                for p in polys.iter().flatten().flatten() {
                    bbox[0] = bbox[0].min(p[0]);
                    bbox[1] = bbox[1].min(p[1]);
                    bbox[2] = bbox[2].max(p[0]);
                    bbox[3] = bbox[3].max(p[1]);
                }
                let x0 = (((bbox[0] - grid.x0) / grid.dx).floor() as i64).clamp(0, grid.nx as i64);
                let x1 = (((bbox[2] - grid.x0) / grid.dx).ceil() as i64).clamp(0, grid.nx as i64);
                let y0 = (((bbox[1] - grid.y0) / grid.dy).floor() as i64).clamp(0, grid.ny as i64);
                let y1 = (((bbox[3] - grid.y0) / grid.dy).ceil() as i64).clamp(0, grid.ny as i64);
                let mut edges = Vec::new();
                for y in y0..y1 {
                    for x in x0..x1 {
                        let west = grid.x0 + x as f64 * grid.dx;
                        let south = grid.y0 + y as f64 * grid.dy;
                        let a = polys
                            .iter()
                            .map(|p| {
                                poly_area(p, Some([west, south, west + grid.dx, south + grid.dy]))
                            })
                            .sum::<f64>();
                        if a > 0. {
                            edges.push((grid.index(x as u32, y as u32), a))
                        }
                    }
                }
                if edges.is_empty() {
                    return Err(format!("divide {s} has no parent grid overlap; projected divide bounds={bbox:?}, parent bounds={:?}",
                        [grid.x0, grid.y0, grid.x0 + grid.nx as f64 * grid.dx,
                         grid.y0 + grid.ny as f64 * grid.dy]).into());
                }
                Ok((edges, total))
            },
    )?;
    let mut out=Weights{offsets:vec![0],indices:vec![],weights:vec![],overlap_m2:vec![],catchment_area_m2:vec![],coverage_fraction:vec![],grid:grid.clone(),method:"projected_polygon_rectangle_area_v1; x-fast; grid.row_order; f64; normalized_covered_area"};
    for (edges, total) in pieces {
        let covered = edges.iter().map(|(_, a)| a).sum::<f64>();
        out.catchment_area_m2.push(total);
        out.coverage_fraction.push(covered / total);
        for (i, a) in edges {
            out.indices.push(i);
            out.overlap_m2.push(a);
            out.weights.push(a / covered)
        }
        out.offsets.push(out.indices.len() as u64);
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concave_split_and_hole_conserve_area() {
        // C shape splits into two pieces when clipped at x=1.5, plus an interior hole.
        let outer = vec![
            [0., 0.],
            [3., 0.],
            [3., 1.],
            [1., 1.],
            [1., 2.],
            [3., 2.],
            [3., 3.],
            [0., 3.],
            [0., 0.],
        ];
        assert_eq!(area(&outer), 7.);
        assert_eq!(clipped_area(&outer, [1.5, 0., 3., 3.]), 3.);
        let hole = vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]];
        let p = vec![outer, hole];
        let sum = (0..3)
            .flat_map(|y| (0..3).map(move |x| (x, y)))
            .map(|(x, y)| poly_area(&p, Some([x as f64, y as f64, x as f64 + 1., y as f64 + 1.])))
            .sum::<f64>();
        assert!((sum - 6.64).abs() < 1e-13);
    }
}

fn lines(blob: &[u8]) -> Result<Vec<Vec<Point>>> {
    if blob.len() < 8 || &blob[..2] != b"GP" || blob[2] != 0 || blob[3] & 0xe0 != 0 {
        return Err("invalid flowpath GeoPackage header".into());
    }
    if blob[3] & 16 != 0 {
        return Ok(vec![]);
    }
    let envelope = match (blob[3] >> 1) & 7 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        _ => return Err("invalid flowpath envelope".into()),
    };
    fn read(c: &mut Cursor<'_>, out: &mut Vec<Vec<Point>>, depth: usize) -> Result<()> {
        if depth > 32 {
            return Err("flowpath WKB nesting exceeds32".into());
        }
        let le = match c.take::<1>()?[0] {
            0 => false,
            1 => true,
            _ => return Err("invalid flowpath WKB byteorder".into()),
        };
        let code = c.u32(le)?;
        let dims = match code / 1000 {
            0 => 2,
            1 | 2 => 3,
            3 => 4,
            _ => return Err("unsupported flowpath EWKB".into()),
        };
        match code % 1000 {
            2 => {
                let n = c.u32(le)? as usize;
                if n > c.b.len() / (8 * dims) {
                    return Err("invalid flowpath point count".into());
                }
                let mut line = Vec::with_capacity(n);
                for _ in 0..n {
                    line.push([c.f64(le)?, c.f64(le)?]);
                    for _ in 2..dims {
                        c.f64(le)?;
                    }
                }
                out.push(line)
            }
            5 | 7 => {
                let n = c.u32(le)? as usize;
                if n > c.b.len() / 5 {
                    return Err("invalid flowpath collection count".into());
                }
                for _ in 0..n {
                    read(c, out, depth + 1)?
                }
            }
            _ => {
                return Err(
                    format!("flowpaths require LineString/MultiLineString, found {code}").into(),
                )
            }
        }
        Ok(())
    }
    let mut c = Cursor {
        b: blob,
        p: 8 + envelope,
    };
    let mut out = Vec::new();
    read(&mut c, &mut out, 0)?;
    if c.p != blob.len() {
        return Err("trailing flowpath WKB bytes".into());
    }
    Ok(out)
}
/// Clip a projected segment to grid and walk intersected cells. Exact corner
/// crossings visit the diagonal cells; edge-only touches use the interior cell.
fn segment_cells(a: Point, b: Point, g: &Grid, out: &mut Vec<u32>) {
    let a = [(a[0] - g.x0) / g.dx, (a[1] - g.y0) / g.dy];
    let b = [(b[0] - g.x0) / g.dx, (b[1] - g.y0) / g.dy];
    let d = [b[0] - a[0], b[1] - a[1]];
    let mut t0 = 0f64;
    let mut t1 = 1f64;
    for (p, q) in [
        (-d[0], a[0]),
        (d[0], g.nx as f64 - a[0]),
        (-d[1], a[1]),
        (d[1], g.ny as f64 - a[1]),
    ] {
        if p == 0. {
            if q < 0. {
                return;
            }
        } else {
            let r = q / p;
            if p < 0. {
                t0 = t0.max(r)
            } else {
                t1 = t1.min(r)
            }
            if t0 > t1 {
                return;
            }
        }
    }
    let start = [a[0] + t0 * d[0], a[1] + t0 * d[1]];
    let end = [a[0] + t1 * d[0], a[1] + t1 * d[1]];
    let mut x = (start[0].floor() as i64).clamp(0, g.nx as i64 - 1);
    let mut y = (start[1].floor() as i64).clamp(0, g.ny as i64 - 1);
    let ex = (end[0].floor() as i64).clamp(0, g.nx as i64 - 1);
    let ey = (end[1].floor() as i64).clamp(0, g.ny as i64 - 1);
    let sx = if d[0] > 0. {
        1
    } else if d[0] < 0. {
        -1
    } else {
        0
    };
    let sy = if d[1] > 0. {
        1
    } else if d[1] < 0. {
        -1
    } else {
        0
    };
    let tx = if sx > 0 {
        ((x + 1) as f64 - a[0]) / d[0]
    } else if sx < 0 {
        (x as f64 - a[0]) / d[0]
    } else {
        f64::INFINITY
    };
    let ty = if sy > 0 {
        ((y + 1) as f64 - a[1]) / d[1]
    } else if sy < 0 {
        (y as f64 - a[1]) / d[1]
    } else {
        f64::INFINITY
    };
    let mut tx = tx;
    let mut ty = ty;
    loop {
        if x < 0 || y < 0 || x >= g.nx as i64 || y >= g.ny as i64 {
            break;
        }
        out.push(g.index(x as u32, y as u32));
        if x == ex && y == ey {
            break;
        }
        if tx < ty {
            x += sx;
            tx += 1. / d[0].abs()
        } else if ty < tx {
            y += sy;
            ty += 1. / d[1].abs()
        } else {
            x += sx;
            y += sy;
            tx += 1. / d[0].abs();
            ty += 1. / d[1].abs();
        }
    }
}
pub struct ChannelRaster {
    pub channel_node: Vec<i32>,
    pub stream_order: Vec<i32>,
}
pub fn rasterize(prepared: &Prepared, grid: &Grid) -> Result<ChannelRaster> {
    let layer = &prepared.layers["flowpaths"];
    let rows = layer
        .rows
        .iter()
        .map(|r| Ok((id(r, "flowpaths")?, r)))
        .collect::<Result<std::collections::HashMap<_, _>>>()?;
    let source = layer
        .crs
        .as_deref()
        .ok_or("geometry CRS missing from gpkg_spatial_ref_sys")?;
    let cells = par_map_projected(
        &prepared.network.node_ids,
        source,
        &grid.crs,
            |projection, s| -> Result<(Vec<u32>, i32)> {
                let row = rows.get(s).ok_or("missing selected flowpath")?;
                let mut ls = lines(row.geometry.as_deref().ok_or("missing flowpath geometry")?)?;
                if let Some(p) = projection {
                    for line in &mut ls {
                        for point in line {
                            *point = p.transform(*point)?
                        }
                    }
                }
                let mut cells = Vec::new();
                for line in ls {
                    for pair in line.windows(2) {
                        segment_cells(pair[0], pair[1], grid, &mut cells)
                    }
                }
                cells.sort_unstable();
                cells.dedup();
                let order = row
                    .properties
                    .get("order")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.);
                if order.fract() != 0. || order < 0. || order > i32::MAX as f64 {
                    return Err("invalid native stream order".into());
                }
                Ok((cells, order as i32))
            },
    )?;
    let mut out = ChannelRaster {
        channel_node: vec![-1; grid.nx as usize * grid.ny as usize],
        stream_order: vec![0; grid.nx as usize * grid.ny as usize],
    };
    // Deterministic junction tie: downstream-most topological rank wins.
    for &node in &prepared.network.ordered_nodes {
        for &cell in &cells[node as usize].0 {
            out.channel_node[cell as usize] = node;
            out.stream_order[cell as usize] = cells[node as usize].1;
        }
    }
    Ok(out)
}
#[cfg(test)]
mod raster_tests {
    use super::*;
    #[test]
    fn projected_metre_units_and_geographic_conversion() {
        assert!(validate_grid_crs("EPSG:5070").is_ok());
        assert!(validate_grid_crs("EPSG:4326").is_err());
        assert!(validate_grid_crs("EPSG:2263").is_err()); // US survey feet
        let p = Projection::new("EPSG:4326", "EPSG:5070").unwrap();
        let origin = p.transform([-96., 23.]).unwrap();
        assert!(origin[0].abs() < 1e-7 && origin[1].abs() < 1e-7);
    }
    #[test]
    fn segments_clip_at_domain_and_cross_grid_corner() {
        let g = Grid {
            nx: 3,
            ny: 3,
            x0: 0.,
            y0: 0.,
            dx: 1.,
            dy: 1.,
            crs: "EPSG:5070".into(),
            row_order: RowOrder::SouthToNorth,
        };
        let mut out = vec![];
        segment_cells([-1., -1.], [4., 4.], &g, &mut out);
        assert_eq!(out, vec![0, 4, 8]);
        out.clear();
        segment_cells([-1., 0.5], [4., 0.5], &g, &mut out);
        assert_eq!(out, vec![0, 1, 2]);
        let north_first = Grid {
            row_order: RowOrder::NorthToSouth,
            ..g.clone()
        };
        out.clear();
        segment_cells([-1., 0.5], [4., 0.5], &north_first, &mut out);
        assert_eq!(out, vec![6, 7, 8]);
        out.clear();
        segment_cells([-1., -1.], [-2., -2.], &g, &mut out);
        assert!(out.is_empty());
    }
}

/// Doctor exercises the same transform used by the data path.
pub fn proj_probe() -> Result<serde_json::Value> {
    let p = Projection::new("EPSG:4326", "EPSG:5070")?.transform([-97., 35.])?;
    if (p[0] + 90515.4982698843).abs() > 0.001 || (p[1] - 1327277.0317004607).abs() > 0.001 {
        return Err(format!("PROJ probe returned unexpected coordinates {p:?}").into());
    }
    Ok(serde_json::json!({"proj_version": hydro_proj::VERSION, "resources": "embedded-only",
        "network": false, "source": "EPSG:4326", "target": "EPSG:5070",
        "input": [-97., 35.], "output": p, "output_bits": p.map(f64::to_bits)}))
}
