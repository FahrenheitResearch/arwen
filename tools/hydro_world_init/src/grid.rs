//! Source grids and catchment area weights.
//!
//! Every divide polygon is moved into the grid's own coordinates (lon/lat for
//! a regular lat/lon grid, the grid CRS for a projected grid), clipped against
//! each axis-aligned cell, and its overlap measured with that grid's area
//! element (exact spherical area for lon/lat, planar for projected). One code
//! path serves every row.
use crate::geom::{self, Polygon};
use crate::proj::Transform;
use crate::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GridSpec {
    /// "regular_latlon" or "projected"
    pub kind: String,
    /// Cell centres as stored in the file (y may run north to south).
    pub y: Vec<f64>,
    pub x: Vec<f64>,
    /// CRS of x/y for a projected grid; "EPSG:4326" for lat/lon.
    pub crs: String,
}

impl GridSpec {
    pub fn nx(&self) -> usize {
        self.x.len()
    }
    pub fn ny(&self) -> usize {
        self.y.len()
    }
    /// Cell edges along one axis, sorted ascending, with the storage index of
    /// each sorted cell.
    fn edges(c: &[f64], clamp_lat: bool) -> Result<(Vec<f64>, Vec<usize>)> {
        if c.len() < 2 {
            return Err("grid axis needs at least two cells".into());
        }
        let mut idx: Vec<usize> = (0..c.len()).collect();
        idx.sort_by(|a, b| c[*a].total_cmp(&c[*b]));
        let s: Vec<f64> = idx.iter().map(|i| c[*i]).collect();
        if s.windows(2).any(|w| w[1] <= w[0]) {
            return Err("grid axis has repeated coordinates".into());
        }
        let mut e = Vec::with_capacity(s.len() + 1);
        e.push(s[0] - 0.5 * (s[1] - s[0]));
        for w in s.windows(2) {
            e.push(0.5 * (w[0] + w[1]));
        }
        let n = s.len();
        e.push(s[n - 1] + 0.5 * (s[n - 1] - s[n - 2]));
        if clamp_lat {
            for v in &mut e {
                *v = v.clamp(-90.0, 90.0);
            }
        }
        Ok((e, idx))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Weights {
    pub schema: String,
    pub fabric: String,
    pub fabric_sha256: String,
    pub grid: GridSpec,
    pub catchment_ids: Vec<String>,
    /// [ny, nx] of the file's 2-D field, so the forcing preparer
    /// (tools/nextgen_forcing, area_weights) reads this file unchanged.
    #[serde(default)]
    pub grid_shape: Vec<usize>,
    /// CSR: weights of catchment c are offsets[c]..offsets[c+1].
    pub offsets: Vec<u64>,
    /// Flat storage index y_index * nx + x_index into the file's 2-D field.
    pub indices: Vec<u32>,
    /// Overlap / covered overlap, summing to one per catchment.
    pub weights: Vec<f64>,
    pub overlap_m2: Vec<f64>,
    pub catchment_area_m2: Vec<f64>,
    pub coverage_fraction: Vec<f64>,
    /// Area-weighted cell-space centroid per catchment (x, y in grid coords).
    pub centroid: Vec<[f64; 2]>,
}

pub const WEIGHTS_SCHEMA: &str = "arwen.hydro-world.init-weights.v1";

/// Shift a lon/lat polygon by whole turns so it sits inside the grid's
/// longitude span (grids may run -180..180 or 0..360 or cross either).
fn wrap_lon(poly: &mut Polygon, x_edges: &[f64]) {
    let (lo, hi) = (x_edges[0], x_edges[x_edges.len() - 1]);
    let pts: Vec<f64> = poly.iter().flatten().map(|p| p[0]).collect();
    let mid = 0.5 * (pts.iter().cloned().fold(f64::INFINITY, f64::min) + pts.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let mut shift = 0.0;
    for k in [-1.0, 1.0, -2.0, 2.0] {
        if !(lo..=hi).contains(&(mid + shift)) && (lo..=hi).contains(&(mid + 360.0 * k)) {
            shift = 360.0 * k;
        }
    }
    if shift != 0.0 {
        for p in poly.iter_mut().flatten() {
            p[0] += shift;
        }
    }
}

pub struct Divides {
    pub ids: Vec<String>,
    pub polys: Vec<Vec<Polygon>>,
    pub area_m2: Vec<f64>,
}

/// Divide polygons and areas from a contract GeoPackage, in file order.
pub fn read_divides(fabric: &hydro_world::contract::Fabric) -> Result<Divides> {
    let rows=fabric.divides.par_iter().map(|d| -> Result<_> {
        let id = d.divide_id.clone().ok_or("divide without divide_id")?;
        let g = d.geom.as_ref().ok_or_else(|| format!("{id}: divide has no geometry"))?;
        let p: Vec<Polygon> = geom::parse_gpkg(&g.0)?
            .into_iter()
            .map(|poly| poly.into_iter().map(geom::open_ring).collect())
            .collect();
        let planar: f64 = p.iter().map(|poly| signed_poly(poly, geom::planar_area)).sum();
        let area=d.areasqkm.filter(|a| a.is_finite() && *a > 0.0).map(|a| a * 1e6).unwrap_or(planar);
        Ok((id,p,area))
    }).collect::<Result<Vec<_>>>()?;
    let mut ids=Vec::with_capacity(rows.len());
    let mut polys=Vec::with_capacity(rows.len());
    let mut area=Vec::with_capacity(rows.len());
    for (id,p,a) in rows { ids.push(id);polys.push(p);area.push(a); }
    Ok(Divides { ids, polys, area_m2: area })
}

fn signed_poly(poly: &Polygon, area: fn(&[geom::Pt]) -> f64) -> f64 {
    let Some(ext) = poly.first() else { return 0.0 };
    let sign = if area(ext) < 0.0 { -1.0 } else { 1.0 };
    poly.iter().map(|r| sign * area(r)).sum()
}

pub fn compute(divides: &Divides, fabric_crs: &str, grid: &GridSpec, densify_m: f64) -> Result<(Vec<(Vec<u32>, Vec<f64>, Vec<f64>, [f64; 2])>, usize)> {
    let latlon = grid.kind == "regular_latlon";
    let (xe, xi) = GridSpec::edges(&grid.x, false)?;
    let (ye, yi) = GridSpec::edges(&grid.y, latlon)?;
    let target = if latlon { "EPSG:4326".to_string() } else { grid.crs.clone() };
    let nx = grid.nx();
    let area_fn: fn(&[geom::Pt]) -> f64 = if latlon { geom::lonlat_area } else { geom::planar_area };
    let (xe, xi, ye, yi, target) = (&xe, &xi, &ye, &yi, &target);
    // One PROJ transform per chunk: creating one costs milliseconds, so it must
    // not happen per divide (rayon's map_init makes one per work split).
    let chunk = divides.polys.len().div_ceil(4 * rayon::current_num_threads()).max(1);
    let rows: Vec<Result<(Vec<u32>, Vec<f64>, Vec<f64>, [f64; 2])>> = divides
        .polys
        .par_chunks(chunk)
        .flat_map_iter(|part| {
            let tr = Transform::new(fabric_crs, &target);
            part.iter().map(move |polys| -> Result<(Vec<u32>, Vec<f64>, Vec<f64>, [f64; 2])> {
                let tr = tr.as_ref().map_err(|e| -> crate::Error { e.to_string().into() })?;
                let mut acc: std::collections::BTreeMap<u32, f64> = Default::default();
                for poly in polys {
                    let mut moved: Polygon = Vec::with_capacity(poly.len());
                    for ring in poly {
                        let dense = geom::densify(ring, densify_m);
                        let mut r = Vec::with_capacity(dense.len());
                        for p in dense {
                            r.push(tr.apply(p[0], p[1])?);
                        }
                        moved.push(r);
                    }
                    if latlon {
                        wrap_lon(&mut moved, &xe);
                    }
                    // Rings may be wound either way: the exterior's winding in grid
                    // space sets the sign, so the exterior adds and holes subtract.
                    let sign = if area_fn(&moved[0]) < 0.0 { -1.0 } else { 1.0 };
                    let (mut bx0, mut bx1, mut by0, mut by1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
                    for p in moved.iter().flatten() {
                        bx0 = bx0.min(p[0]);
                        bx1 = bx1.max(p[0]);
                        by0 = by0.min(p[1]);
                        by1 = by1.max(p[1]);
                    }
                    let j0 = ye.partition_point(|e| *e <= by0).saturating_sub(1);
                    let j1 = ye.partition_point(|e| *e < by1).min(ye.len() - 1);
                    let i0 = xe.partition_point(|e| *e <= bx0).saturating_sub(1);
                    let i1 = xe.partition_point(|e| *e < bx1).min(xe.len() - 1);
                    for j in j0..j1 {
                        let bands: Vec<_> = moved.iter().map(|r| geom::clip_band(r, 1, ye[j], ye[j + 1])).collect();
                        if bands.iter().all(|b| b.len() < 3) {
                            continue;
                        }
                        for i in i0..i1 {
                            let mut a = 0.0;
                            for b in &bands {
                                if b.len() >= 3 {
                                    let c = geom::clip_band(b, 0, xe[i], xe[i + 1]);
                                    a += area_fn(&c);
                                }
                            }
                            let a = a * sign;
                            if a > 0.0 {
                                let flat = (yi[j] * nx + xi[i]) as u32;
                                *acc.entry(flat).or_default() += a;
                            }
                        }
                    }
                }
                let total: f64 = acc.values().sum();
                if total <= 0.0 {
                    return Err("divide does not overlap the source grid".into());
                }
                let mut cx = 0.0;
                let mut cy = 0.0;
                for (k, a) in &acc {
                    let (yy, xx) = ((*k as usize) / nx, (*k as usize) % nx);
                    cx += a * grid.x[xx];
                    cy += a * grid.y[yy];
                }
                let idx: Vec<u32> = acc.keys().copied().collect();
                let ov: Vec<f64> = acc.values().copied().collect();
                let w: Vec<f64> = ov.iter().map(|a| a / total).collect();
                Ok((idx, w, ov, [cx / total, cy / total]))
            }).collect::<Vec<_>>()
        })
        .collect();
    let mut out = Vec::with_capacity(rows.len());
    for (k, r) in rows.into_iter().enumerate() {
        out.push(r.map_err(|e| format!("{}: {e}", divides.ids[k]))?);
    }
    Ok((out, rayon::current_num_threads()))
}

#[cfg(test)]
mod divide_tests {
    use super::*;
    #[test]
    fn parallel_divide_decode_preserves_polygon_axis_and_area_words() {
        let mut fabric=crate::tessellate::build("parallel-divides",30.,-98.,30.2,-97.8,1000.).unwrap();
        // Exercise both the authoritative area and geometry-derived fallback.
        for (i,d) in fabric.divides.iter_mut().enumerate() { if i%2==0 { d.areasqkm=None; } }
        let got=read_divides(&fabric).unwrap();
        for (i,d) in fabric.divides.iter().enumerate() {
            let p:Vec<Polygon>=geom::parse_gpkg(&d.geom.as_ref().unwrap().0).unwrap().into_iter()
                .map(|poly|poly.into_iter().map(geom::open_ring).collect()).collect();
            let planar:f64=p.iter().map(|poly|signed_poly(poly,geom::planar_area)).sum();
            let area=d.areasqkm.filter(|a|a.is_finite() && *a>0.).map(|a|a*1e6).unwrap_or(planar);
            assert_eq!(got.ids[i],d.divide_id.as_ref().unwrap().as_str());
            assert_eq!(got.polys[i],p);
            assert_eq!(got.area_m2[i].to_bits(),area.to_bits());
        }
    }
}
