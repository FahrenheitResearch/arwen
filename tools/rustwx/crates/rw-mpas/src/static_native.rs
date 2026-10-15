//! MPAS-A v8.4.2 static arithmetic. RKIND is binary32; only the integer
//! accumulator division uses R8KIND. These routines intentionally keep each
//! assignment and multiplication in the order used by the pinned Fortran.

use std::collections::VecDeque;

pub const PI: f32 = std::f32::consts::PI;
pub const RAD_PER_DEG: f32 = PI / 180.0;
pub const DEG_PER_RAD: f32 = 180.0 / PI;

/// MPAS: mpas_geotile_manager.F:731-773 (tile_to_latlon).
/// Subsamples start at the source pixel coordinate, with spacing dx/factor;
/// neither a half-pixel offset nor known_x/known_y enters this routine.
pub fn tile_latlon(known_lat: f32, known_lon: f32, dy: f32, dx: f32,
                   xs: i64, ys: i64, i: usize, j: usize, factor: usize) -> (f32, f32) {
    let lat = known_lat + ((ys - 1) * factor as i64 + j as i64) as f32 * dy / factor as f32;
    let lon = known_lon + ((xs - 1) * factor as i64 + i as i64) as f32 * dx / factor as f32;
    (lat * RAD_PER_DEG, lon * RAD_PER_DEG)
}

/// MPAS: mpas_geotile_manager.F:1106-1119 (mpas_latlon_to_xyz).
pub fn xyz(radius: f32, lat: f32, lon: f32) -> [f32; 3] {
    [radius * lon.cos() * lat.cos(), radius * lon.sin() * lat.cos(), radius * lat.sin()]
}

/// MPAS: mpas_kd_tree.F:226-279 (mpas_kd_search_internal).
pub fn distance2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    // Fortran SUM starts at +0, then visits dimensions 1..3.
    ((0.0 + d[0] * d[0]) + d[1] * d[1]) + d[2] * d[2]
}

/// MPAS: mpas_init_atm_static.F:2241-2255 (sphere_distance).
pub fn sphere_distance(lat1: f32, lon1: f32, lat2: f32, lon2: f32, radius: f32) -> f32 {
    let slat = (0.5 * (lat2 - lat1)).sin();
    let slon = (0.5 * (lon2 - lon1)).sin();
    let arg = (slat * slat + (lat1.cos() * lat2.cos()) * (slon * slon)).sqrt();
    2.0 * radius * arg.asin()
}

/// MPAS: mpas_init_atm_static.F:2199-2238 (nearest_cell).
/// Retained native neighbor-walk helper; v8.4.2's driver uses KD search instead.
/// Cell IDs, the starting cell and the result are native one-based IDs.
pub fn nearest_cell(target_lat: f32, target_lon: f32, start_cell: usize,
    max_edges: usize, n_edges: &[usize], neighbors: &[usize], lat: &[f32], lon: &[f32]) -> usize {
    let mut nearest=start_cell;
    let mut current=usize::MAX;
    while nearest!=current {
        current=nearest;
        let mut distance=sphere_distance(lat[current-1],lon[current-1],target_lat,target_lon,1.0);
        for edge in 0..n_edges[current-1] {
            let cell=neighbors[(current-1)*max_edges+edge];
            if cell<=lat.len() {
                let d=sphere_distance(lat[cell-1],lon[cell-1],target_lat,target_lon,1.0);
                if d<distance {nearest=cell;distance=d;}
            }
        }
    }
    nearest
}

/// MPAS: mpas_init_atm_static.F:2258-2289 (max_cell_diameter).
/// Native one-based vertex IDs and cell-major connectivity storage.
pub fn max_cell_diameter(n_cells: usize, max_edges: usize, n_edges: &[usize],
    vertices: &[usize], lat: &[f32], lon: &[f32], vertex_lat: &[f32], vertex_lon: &[f32], radius: f32) -> f32 {
    let mut maximum=0.0f32;
    for cell in 0..n_cells {
        for edge in 0..n_edges[cell] {
            let vertex=vertices[cell*max_edges+edge]-1;
            maximum=maximum.max(sphere_distance(lat[cell],lon[cell],vertex_lat[vertex],vertex_lon[vertex],radius));
        }
    }
    2.0*maximum
}

/// MPAS: src/operators/mpas_geometry_utils.F:131-154 (mpas_arc_length).
pub fn arc_length(a: [f32; 3], b: [f32; 3]) -> f32 {
    let [cx, cy, cz] = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let r = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt();
    let c = (cx*cx + cy*cy + cz*cz).sqrt();
    r * 2.0 * (c / (2.0*r)).asin()
}

/// MPAS: src/operators/mpas_geometry_utils.F:27-72 (mpas_sphere_angle).
pub fn sphere_angle(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let (sa,sb,sc) = (arc_length(b,c),arc_length(a,c),arc_length(a,b));
    let ab=[b[0]-a[0],b[1]-a[1],b[2]-a[2]];
    let ac=[c[0]-a[0],c[1]-a[1],c[2]-a[2]];
    let d=[ab[1]*ac[2]-ab[2]*ac[1],-(ab[0]*ac[2]-ab[2]*ac[0]),ab[0]*ac[1]-ab[1]*ac[0]];
    let s=0.5*(sa+sb+sc);
    let sine=(((s-sb).sin()*(s-sc).sin())/(sb.sin()*sc.sin())).max(0.0).min(1.0).sqrt();
    let angle=2.0*sine.min(1.0).max(-1.0).asin();
    if d[0]*a[0]+d[1]*a[1]+d[2]*a[2] >= 0.0 { angle } else { -angle }
}

/// MPAS: src/operators/mpas_geometry_utils.F:1746-1765 (mpas_rotate_about_vector).
pub fn rotate_about_vector(p: [f32; 3], theta: f32, origin: [f32; 3], axis: [f32; 3]) -> [f32; 3] {
    let [x,y,z]=p;
    let [a,b,c]=origin;
    let [u,v,w]=axis;
    let vw2=v*v+w*w;
    let uw2=u*u+w*w;
    let uv2=u*u+v*v;
    let m=(u*u+v*v+w*w).sqrt();
    let ct=theta.cos();
    let st=theta.sin();
    [
        (a*vw2 + u*(-b*v-c*w+u*x+v*y+w*z) + ((x-a)*vw2+u*(b*v+c*w-v*y-w*z))*ct + m*(-c*v+b*w-w*y+v*z)*st)/(m*m),
        (b*uw2 + v*(-a*u-c*w+u*x+v*y+w*z) + ((y-b)*uw2+v*(a*u+c*w-u*x-w*z))*ct + m*(c*u-a*w+w*x-u*z)*st)/(m*m),
        (c*uv2 + w*(-a*u-b*v+u*x+v*y+w*z) + ((z-c)*uv2+w*(a*u+b*v-u*x-v*y))*ct + m*(-b*u+a*v-v*x+u*y)*st)/(m*m),
    ]
}

/// MPAS: src/operators/mpas_geometry_utils.F:1784-1806 (mpas_mirror_point).
pub fn mirror_point(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    rotate_about_vector(p,2.0*sphere_angle(a,p,b),[0.0;3],a)
}

/// MPAS: src/operators/mpas_geometry_utils.F:1821-1868 (mpas_in_cell).
pub fn in_cell(p: [f32; 3], cell: [f32; 3], vertices: &[[f32; 3]]) -> bool {
    let radius=(cell[0]*cell[0]+cell[1]*cell[1]+cell[2]*cell[2]).sqrt();
    let inv=1.0/radius;
    let distance=arc_length(p,cell);
    for i in 0..vertices.len() {
        let neighbor=mirror_point(cell.map(|v| v*inv),vertices[i].map(|v| v*inv),
                                  vertices[(i+1)%vertices.len()].map(|v| v*inv)).map(|v| v*radius);
        if arc_length(p,neighbor) < distance { return false; }
    }
    true
}

/// MPAS: mpas_init_atm_static.F:1607-1609,1699-1704.
pub fn scaled_mean(sum: i64, count: i32, scale: f32) -> f32 {
    ((sum as f64 / count as f64) as f32) * scale
}

/// MPAS: mpas_init_atm_static.F:1699-1704. The signed count condition
/// precedes the scale, including IEEE signed zero for empty components.
pub fn soilcomp_mean(sum: i64, count: i32, scale: f32) -> f32 {
    (if count > 0 { (sum as f64 / count as f64) as f32 } else { 0.0 }) * scale
}

/// MPAS: mpas_init_atm_static.F:97,1841-1845,1920-1924,2164-2168.
/// MAXLOC compares signed native counts and keeps the first equal maximum.
pub fn dominant_category(counts: &[i32]) -> Option<usize> {
    if counts.is_empty() { return None; }
    let mut best=0;
    for k in 1..counts.len() {
        if counts[k] > counts[best] { best=k; }
    }
    Some(best)
}

/// MPAS: mpas_init_atm_static.F:449-475,1951-1977.
pub fn reconcile_surface(landuse: &mut [i32], soil: &mut [i32], ice: i32,
                         water_lu: i32, water_soil: i32) -> Vec<i32> {
    for (lu, sc) in landuse.iter_mut().zip(soil.iter_mut()) {
        if *lu == ice { *sc = 16; }
        if *lu == water_lu || *sc == water_soil {
            *lu = water_lu;
            *sc = water_soil;
        }
    }
    landuse.iter().map(|&lu| i32::from(lu != water_lu)).collect()
}

/// A single plane with the source's actual halo, in global 1-based indices.
pub struct Plane {
    pub x0: i64,
    pub y0: i64,
    pub nx: usize,
    pub ny: usize,
    pub values: Vec<f32>,
}

impl Plane {
    fn contains(&self, x: i64, y: i64) -> bool {
        x >= self.x0 && y >= self.y0 && x < self.x0 + self.nx as i64 && y < self.y0 + self.ny as i64
    }
    fn slot(&self, x: i64, y: i64) -> usize {
        (y - self.y0) as usize * self.nx + (x - self.x0) as usize
    }
    fn at(&self, x: i64, y: i64) -> f32 { self.values[self.slot(x, y)] }

    /// MPAS: mpas_init_atm_hinterp.F:151-228,794-879 (FOUR_POINT).
    /// Fallback is W_AVERAGE4, W_AVERAGE16, SEARCH, then the missing value.
    pub fn interpolate(&self, x: f32, y: f32, missing: f32) -> f32 {
        let (ix, iy) = (x.floor() as i64, y.floor() as i64);
        let (jx, jy) = (x.ceil() as i64, y.ceil() as i64);
        if self.contains(ix, iy) && self.contains(jx, jy) {
            let (a, b, c, d) = (self.at(ix, iy), self.at(ix, jy), self.at(jx, iy), self.at(jx, jy));
            if [a,b,c,d].iter().all(|&v| v != missing) {
                if ix == jx {
                    return if iy == jy { a } else { a * (jy as f32 - y) + b * (y - iy as f32) };
                }
                if iy == jy { return a * (jx as f32 - x) + c * (x - ix as f32); }
                return (y - iy as f32) * (b * (jx as f32 - x) + d * (x - ix as f32))
                     + (jy as f32 - y) * (a * (jx as f32 - x) + c * (x - ix as f32));
            }
        }
        self.weighted4(x, y, missing)
    }

    /// MPAS: mpas_init_atm_hinterp.F:525-614 (wt_four_pt_average).
    fn weighted4(&self, x: f32, y: f32, missing: f32) -> f32 {
        let (mut ix, mut iy) = (x.floor() as i64, y.floor() as i64);
        let (mut jx, mut jy) = (x.ceil() as i64, y.ceil() as i64);
        let coords = [(ix,iy),(ix,jy),(jx,iy),(jx,jy)];
        let mut w = coords.map(|(i,j)| {
            let (dx,dy) = (x-i as f32,y-j as f32);
            (1.0 - (dx*dx+dy*dy).sqrt()).max(0.0)
        });
        let (xe,ye) = (self.x0+self.nx as i64-1,self.y0+self.ny as i64-1);
        if x > self.x0 as f32-0.5 && ix < self.x0 { ix=self.x0; jx=self.x0; }
        else if x < xe as f32+0.5 && jx > xe { ix=xe; jx=xe; }
        if y > self.y0 as f32-0.5 && iy < self.y0 { iy=self.y0; jy=self.y0; }
        else if y < ye as f32+0.5 && jy > ye { iy=ye; jy=ye; }
        if !self.contains(ix,iy) || !self.contains(jx,jy) { return missing; }
        let v = [self.at(ix,iy),self.at(ix,jy),self.at(jx,iy),self.at(jx,jy)];
        for i in 0..4 { if v[i] == missing { w[i]=0.0; } }
        if w.iter().all(|&v| v == 0.0) { return self.weighted16(x,y,missing); }
        (w[0]*v[0]+w[1]*v[1]+w[2]*v[2]+w[3]*v[3]) / (w[0]+w[1]+w[2]+w[3])
    }

    /// MPAS: mpas_init_atm_hinterp.F:708-786 (wt_sixteen_pt_average).
    fn weighted16(&self, x: f32, y: f32, missing: f32) -> f32 {
        let (ix,iy) = (x.floor() as i64,y.floor() as i64);
        if !self.contains(ix-1,iy-1) || !self.contains(ix+2,iy+2) { return self.search(x,y,missing); }
        let mut sumw=0.0;
        let mut terms=[0.0;16];
        for i in 1..=4 {
            for j in 1..=4 {
                let (px,py)=(ix+3-i,iy+3-j);
                let v=self.at(px,py);
                let (dx,dy)=(x-px as f32,y-py as f32);
                let w=if v==missing {0.0} else {(2.0-(dx*dx+dy*dy).sqrt()).max(0.0)};
                sumw+=w;
                terms[((i-1)*4+j-1) as usize]=w*v;
            }
        }
        if sumw==0.0 { return self.search(x,y,missing); }
        let mut sum=0.0;
        for t in terms { sum+=t; }
        sum/sumw
    }

    /// MPAS: mpas_init_atm_hinterp.F:300-420 (search_extrap).
    /// Queue order is left, right, down, up; finish only the queued frontier
    /// once the first valid point is removed, choosing strictly nearer points.
    fn search(&self, x: f32, y: f32, missing: f32) -> f32 {
        let start=(x.round() as i64,y.round() as i64);
        if !self.contains(start.0,start.1) { return missing; }
        let mut seen=vec![false;self.values.len()];
        let mut queue=VecDeque::from([start]);
        seen[self.slot(start.0,start.1)]=true;
        while let Some((i,j))=queue.pop_front() {
            let v=self.at(i,j);
            for (a,b) in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)] {
                if self.contains(a,b) && !seen[self.slot(a,b)] {
                    seen[self.slot(a,b)]=true;
                    queue.push_back((a,b));
                }
            }
            if v!=missing {
                let mut answer=v;
                let (dx,dy)=(i as f32-x,j as f32-y);
                let mut distance=dx*dx+dy*dy;
                for (a,b) in queue {
                    let value=self.at(a,b);
                    let (dx,dy)=(a as f32-x,b as f32-y);
                    let d=dx*dx+dy*dy;
                    if value!=missing && d<distance { distance=d; answer=value; }
                }
                return answer;
            }
        }
        missing
    }
}
