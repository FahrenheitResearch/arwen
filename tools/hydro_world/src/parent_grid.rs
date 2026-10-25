//! Restrict DEM hydrography to complete cells inside the actual atmosphere grid.
//! Outside cells are open domain boundaries, never unforced catchments.
use crate::{grid::Grid, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ffi::{c_char, c_int, c_void, CString};
type Point = [f64; 2];
#[derive(Clone, Deserialize, Serialize)]
pub struct ParentGrid {
    pub crs: String, pub nx: u32, pub ny: u32,
    pub x0: f64, pub y0: f64, pub dx: f64, pub dy: f64,
}
impl ParentGrid {
    /// Outlet/outline offsets in the source must not create unforced divides.
    /// Keep whole polygons, but require positive land area in the parent.
    pub fn overlaps_divides(&self, divides: &[&[Vec<(f64,f64)>]]) -> Result<Vec<bool>> {
        let bounds = [self.x0, self.y0, self.x0 + self.nx as f64 * self.dx,
                      self.y0 + self.ny as f64 * self.dy];
        let chunks = divides.par_chunks(4096).map(|chunk| -> Result<Vec<bool>> {
            let proj = Projection::new("EPSG:4326", &self.crs)?;
            chunk.iter().map(|rings| {
                let projected = rings.iter().map(|ring| ring.iter().map(|&(x,y)|
                    proj.transform([x,y])).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
                Ok(projected_overlap(&projected, bounds))
            }).collect()
        }).collect::<Result<Vec<_>>>()?;
        Ok(chunks.into_iter().flatten().collect())
    }

    /// Native polygons remain whole; their outlet chooses the owning grid.
    /// Use the projected rectangle, not its larger geographic bounding box.
    pub fn contains_outlets(&self, points: &[(f64, f64)]) -> Result<Vec<bool>> {
        validate_grid_crs(&self.crs)?;
        if self.nx == 0 || self.ny == 0 || self.dx <= 0. || self.dy <= 0. ||
            ![self.x0,self.y0,self.dx,self.dy].iter().all(|x|x.is_finite()) {
            return Err("parent grid needs finite origin and positive metre spacing/extent".into());
        }
        let xmax = self.x0 + self.nx as f64 * self.dx;
        let ymax = self.y0 + self.ny as f64 * self.dy;
        let chunks = points.par_chunks(4096).map(|chunk| -> Result<Vec<bool>> {
            let proj = Projection::new("EPSG:4326", &self.crs)?;
            chunk.iter().map(|&(lon,lat)| {
                let [x,y] = proj.transform([lon,lat])?;
                // Half-open cells give a shared edge exactly one owner.
                Ok(x >= self.x0 && x < xmax && y >= self.y0 && y < ymax)
            }).collect()
        }).collect::<Result<Vec<_>>>()?;
        Ok(chunks.into_iter().flatten().collect())
    }

    pub fn mask(&self, g: &Grid, sea: &mut [bool]) -> Result<usize> {
        validate_grid_crs(&self.crs)?;
        if self.nx == 0 || self.ny == 0 || self.dx <= 0. || self.dy <= 0. ||
            ![self.x0,self.y0,self.dx,self.dy].iter().all(|x|x.is_finite()) {
            return Err("parent grid needs finite origin and positive metre spacing/extent".into());
        }
        let xmax = self.x0 + self.nx as f64 * self.dx;
        let ymax = self.y0 + self.ny as f64 * self.dy;
        // Every worker owns its PROJ context. Require all four cell corners;
        // no catchment can be created solely outside the forced domain.
        let changed: Result<Vec<usize>> = sea.par_chunks_mut(g.nx).enumerate().map(|(r,row)| {
            let proj = Projection::new("EPSG:4326", &self.crs)?;
            let mut n = 0;
            for (c,excluded) in row.iter_mut().enumerate() {
                if *excluded { continue; }
                let mut inside = true;
                for (x,y) in [(c,r),(c+1,r),(c,r+1),(c+1,r+1)] {
                    let (lon,lat) = g.corner(x,y);
                    let [px,py] = proj.transform([lon,lat])?;
                    inside &= px >= self.x0 && px <= xmax && py >= self.y0 && py <= ymax;
                }
                if !inside { *excluded = true; n += 1; }
            }
            Ok(n)
        }).collect();
        Ok(changed?.iter().sum())
    }
}

fn projected_overlap(rings: &[Vec<Point>], bounds: [f64;4]) -> bool {
    // ESRI exteriors and holes have opposite orientation. Preserve that sign
    // while clipping; an atmosphere rectangle entirely in a hole has zero area.
    let signed: f64 = rings.iter().map(|ring| {
        let mut clipped = ring.clone();
        for (axis, edge, greater) in [(0,bounds[0],true),(0,bounds[2],false),
                                     (1,bounds[1],true),(1,bounds[3],false)] {
            let mut out = Vec::new();
            if let Some(&last) = clipped.last() {
                let mut a = last;
                for &b in &clipped {
                    let ia = if greater {a[axis] >= edge} else {a[axis] <= edge};
                    let ib = if greater {b[axis] >= edge} else {b[axis] <= edge};
                    if ia != ib {
                        let f = (edge-a[axis])/(b[axis]-a[axis]);
                        let mut p = [a[0]+f*(b[0]-a[0]),a[1]+f*(b[1]-a[1])];
                        p[axis] = edge;
                        out.push(p);
                    }
                    if ib {out.push(b);}
                    a = b;
                }
            }
            clipped = out;
        }
        if clipped.len() < 3 {return 0.;}
        let origin = clipped[0];
        clipped.iter().zip(clipped.iter().cycle().skip(1)).map(|(a,b)|
            (a[0]-origin[0])*(b[1]-origin[1])-(b[0]-origin[0])*(a[1]-origin[1])
        ).sum::<f64>() / 2.
    }).sum();
    signed.abs() > 0.
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

#[cfg(test)]
mod native_outlet_tests {
    use super::*;

    #[test]
    fn native_polygon_outside_parent_despite_inset_flowline_outlet() {
        let b = [-2352000.,-303773.0349141713,2352000.,2504226.9650858287];
        // Actual cat-2495519 outline begins 15.77 m north of the HRRR recipe
        // grid, although its source flowline endpoint falls inside the grid.
        let r = vec![[-1966709.,2504242.7338465466],[-1963978.,2504242.7338465466],
                     [-1963978.,2507164.],[-1966709.,2507164.],[-1966709.,2504242.7338465466]];
        assert!(!projected_overlap(&[r], b));
        let outer = vec![[-1.,-1.],[-1.,1001.],[1001.,1001.],[1001.,-1.],[-1.,-1.]];
        let hole = vec![[-0.5,-0.5],[1000.5,-0.5],[1000.5,1000.5],[-0.5,1000.5],[-0.5,-0.5]];
        assert!(projected_overlap(&[outer.clone()], [0.,0.,1000.,1000.]));
        assert!(!projected_overlap(&[outer,hole], [0.,0.,1000.,1000.]));
        let crossing = vec![[-1.,0.5],[0.5,2.],[2.,0.5],[0.5,-1.],[-1.,0.5]];
        assert!(projected_overlap(&[crossing], [0.,0.,1.,1.]));
    }

    #[test]
    fn native_outlets_use_projected_parent_not_geographic_bbox() {
        let grid = ParentGrid {
            crs: "+proj=lcc +lat_1=27.19 +lat_2=47.19 +lat_0=27.19 +lon_0=-97.5063 +a=6370000 +b=6370000 +units=m +no_defs +type=crs".into(),
            nx: 392, ny: 234, x0: -2352000., y0: -303773.0349141713,
            dx: 12000., dy: 12000.,
        };
        // Both points are in this parent's geographic bounding rectangle.
        // The northwest one lies outside its curved projected footprint.
        let mut points = vec![(-97.5063,37.18615);8193];
        points[4096] = (-128.,49.);
        let inside = grid.contains_outlets(&points).unwrap();
        assert_eq!(inside.len(),points.len());
        assert!(!inside[4096]);
        assert_eq!(inside.iter().filter(|&&x|x).count(),8192);
        let flat = ParentGrid { crs: "EPSG:3857".into(), nx: 1, ny: 1,
            x0: 0., y0: 0., dx: 1000., dy: 1000. };
        assert_eq!(flat.contains_outlets(&[(0.,0.),(-0.01,0.),(0.01,0.)]).unwrap(),
                   vec![true,false,false]);
    }
}
