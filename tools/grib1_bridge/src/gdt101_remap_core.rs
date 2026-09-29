//! Dependency-free numerical/wire core for GDT-101 source normalization.
//!
//! WMO grid-definition template 101 is an UNSTRUCTURED mesh: the grid
//! definition carries a cell count and a mesh identity, and the cell
//! coordinates travel in separate records.  Nothing here knows which model
//! produced the mesh; the caller supplies its identity and the window to
//! remap onto.  Production field decoding remains grib-core's, not a second
//! implementation.
use std::io::{Read, Write};

pub const CONTRACT: &str = "arwen.gdt101-regional-remap.v1";
pub const MAX_TARGET: usize = 2_000_000;
pub const MAX_SOURCE: usize = 20_000_000;
// P3 adds the unpublished-cell inventory after the stencils; a P2 plan has
// no such list and is refused rather than read as a mesh with none.
const PLAN_MAGIC: &[u8; 8] = b"GDT101P3";
const EARTH_RADIUS_M: f64 = 6_371_229.;
pub type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    pub west: f64,
    pub south: f64,
    pub dx: f64,
    pub dy: f64,
    pub nx: usize,
    pub ny: usize,
}
impl Target {
    pub fn validate(&self) -> Result<()> {
        if ![self.west, self.south, self.dx, self.dy].iter().all(|x| x.is_finite()) {
            return Err("nonfinite target geometry".into());
        }
        // The SPACING is the caller's declared product choice, not this
        // crate's: it checks only that the window is a usable regular grid.
        if !(self.dx > 0. && self.dx <= 1. && self.dy > 0. && self.dy <= 1.)
            || self.nx < 2 || self.ny < 2
        {
            return Err("target must be at least 2x2 at a positive spacing no coarser than one degree".into());
        }
        if self.nx.checked_mul(self.ny).map_or(true, |n| n > MAX_TARGET)
            || !(-180.0..180.0).contains(&self.west)
            || (self.nx - 1) as f64 * self.dx >= 180.
            || self.south < -88.
            || self.south + (self.ny - 1) as f64 * self.dy > 88.
        {
            return Err("target exceeds the regional, non-polar or memory envelope".into());
        }
        Ok(())
    }
    pub fn len(&self) -> usize { self.nx * self.ny }
    pub fn parse(text: &str) -> Result<Self> {
        let p: Vec<_> = text.split_whitespace().collect();
        if p.len() != 6 { return Err("target must be west south dx dy nx ny".into()); }
        let value = Self {
            west: p[0].parse().map_err(|_| "bad west")?,
            south: p[1].parse().map_err(|_| "bad south")?,
            dx: p[2].parse().map_err(|_| "bad dx")?,
            dy: p[3].parse().map_err(|_| "bad dy")?,
            nx: p[4].parse().map_err(|_| "bad nx")?,
            ny: p[5].parse().map_err(|_| "bad ny")?,
        };
        value.validate()?;
        Ok(value)
    }
}

pub fn xyz(lat: f64, lon: f64) -> Result<[f64; 3]> {
    if !lat.is_finite() || !lon.is_finite() || lat.abs() > 90. || lon.abs() > 360. {
        return Err("source coordinates must be finite geographical degrees".into());
    }
    let (s, c) = lat.to_radians().sin_cos();
    let (sl, cl) = lon.to_radians().sin_cos();
    Ok([c * cl, c * sl, s])
}
#[derive(Clone, Debug)]
struct Point { xyz: [f64; 3], id: u32 }
#[derive(Clone, Copy, Debug)]
struct Candidate { d2: f64, id: u32 }
const EMPTY: Candidate = Candidate { d2: f64::INFINITY, id: u32::MAX };

fn distance2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}
fn partition(points: &mut [Point], depth: usize) {
    if points.is_empty() { return; }
    let axis = depth % 3;
    let mid = points.len() / 2;
    points.select_nth_unstable_by(mid, |a, b| {
        a.xyz[axis].total_cmp(&b.xyz[axis]).then(a.id.cmp(&b.id))
    });
    let (left, right) = points.split_at_mut(mid);
    partition(left, depth + 1);
    partition(&mut right[1..], depth + 1);
}
fn insert(best: &mut [Candidate; 4], candidate: Candidate) {
    let mut position = 4;
    for (i, old) in best.iter().enumerate() {
        if candidate.d2 < old.d2 || (candidate.d2 == old.d2 && candidate.id < old.id) {
            position = i;
            break;
        }
    }
    if position < 4 {
        for i in (position + 1..4).rev() { best[i] = best[i - 1]; }
        best[position] = candidate;
    }
}
fn search(points: &[Point], q: &[f64; 3], depth: usize, best: &mut [Candidate; 4]) {
    if points.is_empty() { return; }
    let mid = points.len() / 2;
    let point = &points[mid];
    insert(best, Candidate { d2: distance2(q, &point.xyz), id: point.id });
    let delta = q[depth % 3] - point.xyz[depth % 3];
    let (near, far) = if delta <= 0. {
        (&points[..mid], &points[mid + 1..])
    } else {
        (&points[mid + 1..], &points[..mid])
    };
    search(near, q, depth + 1, best);
    // <= preserves deterministic index-order ties across the split plane.
    if delta * delta <= best[3].d2 { search(far, q, depth + 1, best); }
}

#[derive(Clone, Debug)]
pub struct Stencil {
    pub ids: [u32; 4],
    pub weights: [f64; 4],
    // Bit i is the native FR_LAND >= 0.5 class of ids[i]. The nearest
    // donor (ids[0]) owns the target land/water class and whole soil column.
    pub land_bits: u8,
}
#[derive(Clone, Debug)]
pub struct Plan {
    pub source_count: usize,
    /// Originating centre (Section 1) every field of this mesh must declare.
    pub centre: u16,
    pub source_grid: Vec<u8>,
    pub target: Target,
    pub stencils: Vec<Stencil>,
    /// Cells the land-fraction record leaves missing, ascending.  A
    /// limited-area mesh publishes its lateral boundary strip as missing in
    /// every record (DWD's regional ICON masks 16,968 of 542,040 cells), so
    /// those cells have no land/water class and no values: they are never
    /// donors, and a field that publishes a value at one of them is refused
    /// in `apply`, because the plan could not classify that cell.  Empty for
    /// a mesh whose land fraction is complete, which is every global mesh.
    pub unpublished: Vec<u32>,
}
impl Plan {
    pub fn validate(&self) -> Result<()> {
        self.target.validate()?;
        if !(4..=MAX_SOURCE).contains(&self.source_count)
            || self.stencils.len() != self.target.len()
            || self.source_grid.len() < 30 || self.source_grid.len() > 256
            || self.source_grid[4] != 3
            || u16::from_be_bytes([self.source_grid[12], self.source_grid[13]]) != 101
            || u32::from_be_bytes(self.source_grid[6..10].try_into().unwrap()) as usize != self.source_count
            || u32::from_be_bytes(self.source_grid[0..4].try_into().unwrap()) as usize != self.source_grid.len()
        {
            return Err("invalid plan source grid, count or stencil inventory".into());
        }
        if self.unpublished.len() + 4 > self.source_count
            || self.unpublished.windows(2).any(|pair| pair[0] >= pair[1])
            || self.unpublished.last().map_or(false, |&id| id as usize >= self.source_count)
        {
            return Err("invalid unpublished-cell inventory".into());
        }
        for s in &self.stencils {
            if s.land_bits > 15 || s.ids.iter().any(|&id| id as usize >= self.source_count)
                || s.weights.iter().any(|&w| !w.is_finite() || w < 0. || w > 1.)
                || (s.weights.iter().sum::<f64>() - 1.).abs() > 1e-12
            {
                return Err("invalid donor index, class or normalized weight".into());
            }
            for i in 0..4 {
                for j in i + 1..4 {
                    if s.ids[i] == s.ids[j] { return Err("duplicate donor in plan".into()); }
                }
            }
            if s.ids.iter().any(|id| self.unpublished.binary_search(id).is_ok()) {
                return Err("an unpublished cell is a donor in the plan".into());
            }
        }
        Ok(())
    }
    pub fn write<W: Write>(&self, mut w: W) -> Result<()> {
        self.validate()?;
        w.write_all(PLAN_MAGIC).map_err(|e| e.to_string())?;
        put_u32(&mut w, self.source_count as u32)?;
        put_u32(&mut w, self.centre as u32)?;
        put_u32(&mut w, self.source_grid.len() as u32)?;
        w.write_all(&self.source_grid).map_err(|e| e.to_string())?;
        for x in [self.target.west, self.target.south, self.target.dx, self.target.dy] {
            w.write_all(&x.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        put_u32(&mut w, self.target.nx as u32)?;
        put_u32(&mut w, self.target.ny as u32)?;
        for s in &self.stencils {
            for id in s.ids { put_u32(&mut w, id)?; }
            for x in s.weights { w.write_all(&x.to_le_bytes()).map_err(|e| e.to_string())?; }
            w.write_all(&[s.land_bits]).map_err(|e| e.to_string())?;
        }
        put_u32(&mut w, self.unpublished.len() as u32)?;
        for &id in &self.unpublished { put_u32(&mut w, id)?; }
        w.flush().map_err(|e| e.to_string())
    }
    pub fn read<R: Read>(mut r: R) -> Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != PLAN_MAGIC { return Err("unknown remap plan version".into()); }
        let source_count = get_u32(&mut r)? as usize;
        if !(4..=MAX_SOURCE).contains(&source_count) { return Err("invalid plan source count".into()); }
        let centre = get_u32(&mut r)?;
        if centre > u16::MAX as u32 { return Err("invalid plan originating centre".into()); }
        let centre = centre as u16;
        let ngrid = get_u32(&mut r)? as usize;
        if !(30..=256).contains(&ngrid) { return Err("invalid plan grid identity length".into()); }
        let mut source_grid = vec![0; ngrid];
        r.read_exact(&mut source_grid).map_err(|e| e.to_string())?;
        let target = Target {
            west: get_f64(&mut r)?, south: get_f64(&mut r)?,
            dx: get_f64(&mut r)?, dy: get_f64(&mut r)?,
            nx: get_u32(&mut r)? as usize, ny: get_u32(&mut r)? as usize,
        };
        target.validate()?;
        let mut stencils = Vec::with_capacity(target.len());
        for _ in 0..target.len() {
            let mut ids = [0; 4];
            let mut weights = [0.; 4];
            for id in &mut ids { *id = get_u32(&mut r)?; }
            for x in &mut weights { *x = get_f64(&mut r)?; }
            let mut flag = [0];
            r.read_exact(&mut flag).map_err(|e| e.to_string())?;
            stencils.push(Stencil { ids, weights, land_bits: flag[0] });
        }
        let count = get_u32(&mut r)? as usize;
        if count + 4 > source_count { return Err("invalid unpublished-cell count".into()); }
        let mut unpublished = Vec::with_capacity(count);
        for _ in 0..count { unpublished.push(get_u32(&mut r)?); }
        let mut extra = [0];
        if r.read(&mut extra).map_err(|e| e.to_string())? != 0 {
            return Err("trailing bytes after the remap plan".into());
        }
        let plan = Self { source_count, centre, source_grid, target, stencils, unpublished };
        plan.validate()?;
        Ok(plan)
    }
}
fn put_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())
}
fn get_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0; 4]; r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}
fn get_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut b = [0; 8]; r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(f64::from_le_bytes(b))
}

pub fn build_plan(lat: &[f64], lon: &[f64], land: &[f64], source_grid: Vec<u8>,
                  centre: u16, target: Target, max_distance_m: f64) -> Result<Plan> {
    target.validate()?;
    if lat.len() != lon.len() || lat.len() != land.len()
        || !(4..=MAX_SOURCE).contains(&lat.len())
        || !max_distance_m.is_finite() || max_distance_m <= 0. || max_distance_m > 1_000_000.
    {
        return Err("invalid source array lengths or search radius".into());
    }
    if land.iter().any(|&x| x.is_infinite() || (x.is_finite() && !(0.0..=1.0).contains(&x))) {
        return Err("native land fractions are outside [0,1]".into());
    }
    // A cell whose land fraction is missing is a cell the producer does not
    // publish (a limited-area mesh's lateral boundary strip).  It has no
    // land/water class for the class-aware methods to read, so it is left
    // out of the donor search entirely and recorded, and `apply` refuses a
    // field that publishes a value there.  The coordinates are still read
    // for every cell: they travel complete in their own records.
    let mut points = Vec::with_capacity(lat.len());
    let mut unpublished = Vec::new();
    for i in 0..lat.len() {
        let position = xyz(lat[i], lon[i])?;
        if land[i].is_nan() { unpublished.push(i as u32); continue; }
        points.push(Point { xyz: position, id: i as u32 });
    }
    if points.len() < 4 {
        return Err("fewer than four native cells carry a land fraction".into());
    }
    partition(&mut points, 0);
    let max_d2 = (2. * (max_distance_m / EARTH_RADIUS_M / 2.).sin()).powi(2);
    let mut stencils = Vec::with_capacity(target.len());
    for j in 0..target.ny {
        for i in 0..target.nx {
            let q = xyz(target.south + j as f64 * target.dy,
                        (target.west + i as f64 * target.dx + 180.).rem_euclid(360.) - 180.)?;
            let mut best = [EMPTY; 4];
            search(&points, &q, 0, &mut best);
            if best[3].d2 > max_d2 || best[3].id == u32::MAX {
                return Err(format!("four native donors do not reach target i={i}, j={j} within {max_distance_m} m"));
            }
            let mut weights = [0.; 4];
            if best[0].d2 <= 1e-24 { weights[0] = 1.; }
            else {
                let sum: f64 = best.iter().map(|p| 1. / p.d2).sum();
                for n in 0..4 { weights[n] = (1. / best[n].d2) / sum; }
            }
            let ids = [best[0].id, best[1].id, best[2].id, best[3].id];
            let mut land_bits = 0;
            for (n, &id) in ids.iter().enumerate() {
                if land[id as usize] >= 0.5 { land_bits |= 1 << n; }
            }
            stencils.push(Stencil { ids, weights, land_bits });
        }
    }
    let plan = Plan { source_count: lat.len(), centre, source_grid, target, stencils, unpublished };
    plan.validate()?;
    Ok(plan)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method { Idw4, SurfaceIdw4, Nearest, SoilNearest, SeaIceNearest }
impl Method {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "idw4" => Ok(Self::Idw4), "surface-idw4" => Ok(Self::SurfaceIdw4),
            "nearest" => Ok(Self::Nearest), "soil-nearest" => Ok(Self::SoilNearest),
            "seaice-nearest" => Ok(Self::SeaIceNearest),
            _ => Err(format!("unknown interpolation method {s:?}")),
        }
    }
}
pub fn apply(plan: &Plan, field: &[f64], method: Method) -> Result<Vec<f64>> {
    plan.validate()?;
    if field.len() != plan.source_count { return Err("field/plan point-count mismatch".into()); }
    if let Some(&id) = plan.unpublished.iter().find(|&&id| field[id as usize].is_finite()) {
        return Err(format!(
            "field publishes native cell {id}, which the land-fraction record leaves missing; \
             the plan has no land/water class for it"));
    }
    let mut out = Vec::with_capacity(plan.target.len());
    for (index, s) in plan.stencils.iter().enumerate() {
        let target_land = s.land_bits & 1 != 0;
        let value = match method {
            Method::SoilNearest if !target_land => f64::NAN,
            Method::SeaIceNearest if target_land => 0.,
            Method::Nearest | Method::SoilNearest | Method::SeaIceNearest => {
                let x = field[s.ids[0] as usize];
                if !x.is_finite() { return Err(format!("missing nearest donor at target {index}")); }
                if method == Method::SeaIceNearest && !(0.0..=1.0).contains(&x) {
                    return Err("sea ice fraction outside [0,1]".into());
                }
                x
            },
            Method::Idw4 | Method::SurfaceIdw4 => {
                let mut value = 0.;
                let mut mass = 0.;
                for n in 0..4 {
                    let weight = s.weights[n];
                    if weight == 0. { continue; }
                    if method == Method::SurfaceIdw4 && ((s.land_bits & (1 << n)) != 0) != target_land {
                        continue;
                    }
                    let x = field[s.ids[n] as usize];
                    // Missing positive-weight donors are not silently dropped.
                    if !x.is_finite() { return Err(format!("missing positive-weight donor at target {index}")); }
                    value += weight * x;
                    mass += weight;
                }
                if mass <= 0. || !mass.is_finite() { return Err("empty remap support".into()); }
                let x = value / mass;
                if !x.is_finite() { return Err("nonfinite remapped field".into()); }
                x
            },
        };
        out.push(value);
    }
    Ok(out)
}

/// A strict single-field GRIB2 envelope view. Multi-field or concatenated
/// objects are refused here because each source object promises one field.
pub struct Envelope<'a> { pub discipline: u8, pub sections: [Option<&'a [u8]>; 8] }
pub fn envelope(data: &[u8]) -> Result<Envelope<'_>> {
    if data.len() < 20 || &data[..4] != b"GRIB" || data[7] != 2
        || u64::from_be_bytes(data[8..16].try_into().unwrap()) != data.len() as u64
        || &data[data.len() - 4..] != b"7777"
    {
        return Err("not one complete GRIB2 envelope".into());
    }
    let mut sections = [None; 8];
    let mut offset = 16;
    let mut previous = 0;
    while offset < data.len() - 4 {
        if offset + 5 > data.len() - 4 { return Err("truncated section header".into()); }
        let n = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        let number = data[offset + 4] as usize;
        if n < 5 || offset.checked_add(n).map_or(true, |end| end > data.len() - 4)
            || number == 0 || number > 7 || number <= previous || sections[number].is_some()
        {
            return Err("invalid, repeated or out-of-order GRIB2 section".into());
        }
        sections[number] = Some(&data[offset..offset + n]);
        previous = number;
        offset += n;
    }
    for (n, min_len) in [(1, 21), (3, 14), (4, 34), (5, 11), (6, 6), (7, 5)] {
        if sections[n].map_or(true, |s| s.len() < min_len) {
            return Err(format!("missing or short GRIB2 section {n}"));
        }
    }
    Ok(Envelope { discipline: data[6], sections })
}
fn section(number: u8, length: usize) -> Vec<u8> {
    let mut s = vec![0; length];
    s[..4].copy_from_slice(&(length as u32).to_be_bytes());
    s[4] = number;
    s
}
fn sign_magnitude(degrees: f64) -> u32 {
    let magnitude = (degrees.abs() * 1_000_000.).round() as u32;
    magnitude | if degrees < 0. { 0x8000_0000 } else { 0 }
}
fn longitude(degrees: f64) -> u32 { (degrees.rem_euclid(360.) * 1_000_000.).round() as u32 }
pub fn regular_grid(target: Target) -> Result<Vec<u8>> {
    target.validate()?;
    let mut s = section(3, 72);
    s[6..10].copy_from_slice(&(target.len() as u32).to_be_bytes());
    s[14] = 6; // WMO Table 3.2: sphere of radius 6,371,229 m.
    s[15..30].fill(255); // Unused explicit radius/ellipsoid scalings.
    s[30..34].copy_from_slice(&(target.nx as u32).to_be_bytes());
    s[34..38].copy_from_slice(&(target.ny as u32).to_be_bytes());
    s[42..46].fill(255); // Basic angle zero: standard microdegrees.
    s[46..50].copy_from_slice(&sign_magnitude(target.south).to_be_bytes());
    s[50..54].copy_from_slice(&longitude(target.west).to_be_bytes());
    s[54] = 0x30; // Both increments present; U/V remain Earth-relative.
    s[55..59].copy_from_slice(&sign_magnitude(target.south + (target.ny - 1) as f64 * target.dy).to_be_bytes());
    s[59..63].copy_from_slice(&longitude(target.west + (target.nx - 1) as f64 * target.dx).to_be_bytes());
    s[63..67].copy_from_slice(&((target.dx * 1_000_000.).round() as u32).to_be_bytes());
    s[67..71].copy_from_slice(&((target.dy * 1_000_000.).round() as u32).to_be_bytes());
    s[71] = 0x40; // +i east, +j north, row-major, no alternating rows.
    Ok(s)
}

/// Preserve identification, local metadata and the complete product section;
/// replace only the horizontal grid, packing and values. IEEE f32 output
/// avoids another lossy packing tolerance. Missing soil on water uses a bitmap.
pub fn encode_regular(template: &[u8], target: Target, values: &[f64]) -> Result<Vec<u8>> {
    let e = envelope(template)?;
    target.validate()?;
    if values.len() != target.len() { return Err("output field/target count mismatch".into()); }
    if values.iter().any(|x| x.is_infinite() || (x.is_finite() && !(*x as f32).is_finite())) {
        return Err("field overflows finite IEEE f32 encoding".into());
    }
    let present = values.iter().filter(|x| x.is_finite()).count();
    let s3 = regular_grid(target)?;
    let mut s5 = if present == 0 { section(5, 21) } else { section(5, 12) };
    s5[5..9].copy_from_slice(&(present as u32).to_be_bytes());
    if present != 0 { s5[9..11].copy_from_slice(&4u16.to_be_bytes()); s5[11] = 1; }
    // grib-core's all-missing field contract is explicit all-zero bitmap +
    // empty simple-packed data, rather than an invalid zero-count IEEE field.
    let mut s6 = if present == values.len() { section(6, 6) }
                 else { section(6, 6 + (values.len() + 7) / 8) };
    if present == values.len() { s6[5] = 255; }
    else {
        for (i, value) in values.iter().enumerate() {
            if value.is_finite() { s6[6 + i / 8] |= 1 << (7 - i % 8); }
        }
    }
    let mut s7 = section(7, 5 + present * 4);
    let mut offset = 5;
    for value in values.iter().filter(|x| x.is_finite()) {
        s7[offset..offset + 4].copy_from_slice(&(*value as f32).to_be_bytes());
        offset += 4;
    }
    let mut out = vec![0; 16];
    out[..4].copy_from_slice(b"GRIB");
    out[6] = e.discipline; out[7] = 2;
    out.extend_from_slice(e.sections[1].unwrap());
    if let Some(local) = e.sections[2] { out.extend_from_slice(local); }
    out.extend_from_slice(&s3);
    out.extend_from_slice(e.sections[4].unwrap());
    out.extend_from_slice(&s5); out.extend_from_slice(&s6); out.extend_from_slice(&s7);
    out.extend_from_slice(b"7777");
    let length = out.len() as u64;
    out[8..16].copy_from_slice(&length.to_be_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn target() -> Target { Target { west: 0., south: 0., dx: 0.125, dy: 0.125, nx: 2, ny: 2 } }
    fn native_grid(n: usize) -> Vec<u8> {
        let mut s = section(3, 35);
        s[6..10].copy_from_slice(&(n as u32).to_be_bytes());
        s[12..14].copy_from_slice(&101u16.to_be_bytes()); s
    }
    fn plan() -> Plan {
        build_plan(&[0.,0.,0.125,0.125], &[0.,0.125,0.,0.125], &[1.,0.,1.,0.],
                   native_grid(4), 78, target(), 80_000.).unwrap()
    }
    fn template() -> Vec<u8> {
        let mut s = vec![0;16]; s[..4].copy_from_slice(b"GRIB"); s[7]=2;
        for part in [section(1,21), native_grid(4), section(4,34), section(5,21), section(6,6), section(7,5)] {
            s.extend_from_slice(&part);
        }
        s.extend_from_slice(b"7777"); let n=s.len() as u64;
        s[8..16].copy_from_slice(&n.to_be_bytes()); s
    }
    #[test] fn dateline_xyz_is_continuous() {
        assert!(distance2(&xyz(10.,179.999).unwrap(), &xyz(10.,-179.999).unwrap()) < 2e-9);
    }
    #[test] fn longitude_branches_agree() { assert!(distance2(&xyz(0.,-10.).unwrap(), &xyz(0.,350.).unwrap()) < 1e-25); }
    #[test] fn bad_geometry_is_refused() {
        assert!(xyz(f64::NAN,0.).is_err()); assert!(xyz(91.,0.).is_err());
        assert!(Target { nx: usize::MAX, ..target() }.validate().is_err());
        assert!(Target { south: 88., ..target() }.validate().is_err());
        assert!(Target { dx: 0., ..target() }.validate().is_err());
        assert!(Target { dy: 1.5, ..target() }.validate().is_err());
    }
    #[test] fn the_plan_carries_the_mesh_identity_and_its_centre() {
        let p = plan();
        assert_eq!((p.centre, p.source_count), (78, 4));
        let mut bytes = Vec::new(); p.write(&mut bytes).unwrap();
        assert_eq!(Plan::read(&bytes[..]).unwrap().centre, 78);
    }
    #[test] fn nearest_exact_values_and_scan_order() {
        assert_eq!(apply(&plan(), &[1.,2.,3.,4.], Method::Idw4).unwrap(), vec![1.,2.,3.,4.]);
    }
    #[test] fn constants_are_preserved() {
        let mut t=target(); t.west=0.03;t.south=0.03;
        let p=build_plan(&[0.,0.,0.25,0.25], &[0.,0.25,0.,0.25], &[1.;4], native_grid(4),78,t,80_000.).unwrap();
        for x in apply(&p,&[273.15;4],Method::Idw4).unwrap() { assert!((x-273.15).abs()<1e-12); }
    }
    #[test] fn weights_form_convex_combinations() {
        let mut t=target();t.west=0.04;t.south=0.04;
        let p=build_plan(&[0.,0.,0.25,0.25], &[0.,0.25,0.,0.25], &[1.;4], native_grid(4),78,t,80_000.).unwrap();
        for x in apply(&p,&[-2.,0.,4.,8.],Method::Idw4).unwrap() { assert!((-2.0..=8.0).contains(&x)); }
    }
    #[test] fn water_soil_is_missing_and_land_column_is_nearest() {
        let out=apply(&plan(), &[270.,f64::NAN,280.,f64::NAN],Method::SoilNearest).unwrap();
        assert_eq!(out[0],270.);assert!(out[1].is_nan());assert_eq!(out[2],280.);assert!(out[3].is_nan());
    }
    #[test] fn missing_land_soil_is_not_invented() { assert!(apply(&plan(), &[f64::NAN;4],Method::SoilNearest).is_err()); }
    #[test] fn seaice_is_zero_on_land_and_kept_on_water() {
        assert_eq!(apply(&plan(), &[f64::NAN,0.7,f64::NAN,0.2],Method::SeaIceNearest).unwrap(), vec![0.,0.7,0.,0.2]);
    }
    #[test] fn missing_positive_atmospheric_weight_fails() {
        let mut p=plan();p.stencils[0].weights=[0.25;4];
        assert!(apply(&p,&[1.,2.,f64::NAN,4.],Method::Idw4).is_err());
    }
    #[test] fn surface_does_not_blend_land_and_water() {
        let mut p=plan();p.stencils[0].weights=[0.25;4];
        let s=&p.stencils[0];let mut values=vec![0.;4];
        for n in 0..4 {values[s.ids[n] as usize]=if s.land_bits & (1<<n)!=0 {280.} else {300.};}
        assert!((apply(&p,&values,Method::SurfaceIdw4).unwrap()[0]-280.).abs()<1e-12);
    }
    #[test] fn plan_roundtrip_and_truncation() {
        let p=plan();let mut bytes=Vec::new();p.write(&mut bytes).unwrap();
        let q=Plan::read(&bytes[..]).unwrap();assert_eq!(q.target,p.target);
        assert!(Plan::read(&bytes[..bytes.len()-1]).is_err());
        bytes.push(0);assert!(Plan::read(&bytes[..]).is_err());
    }
    #[test] fn damaged_weights_are_rejected() {let mut p=plan();p.stencils[0].weights[0]=f64::NAN;assert!(p.validate().is_err());}
    #[test] fn source_grid_count_cannot_drift() {let mut p=plan();p.source_count=5;assert!(p.validate().is_err());}
    /// A limited-area mesh: cell 0 sits exactly on a target point but is
    /// missing from the land-fraction record, the way a lateral boundary
    /// strip is published.
    fn masked_mesh() -> Plan {
        build_plan(&[0.,0.,0.125,0.125,0.0625,0.25], &[0.,0.125,0.,0.125,0.0625,0.25],
                   &[f64::NAN,1.,0.,1.,0.,1.], native_grid(6),78,target(),80_000.).unwrap()
    }
    #[test] fn unpublished_cells_are_recorded_and_never_donors() {
        let p=masked_mesh();
        assert_eq!(p.unpublished, vec![0]);
        assert!(p.stencils.iter().all(|s| !s.ids.contains(&0)));
        let mut bytes=Vec::new(); p.write(&mut bytes).unwrap();
        assert_eq!(Plan::read(&bytes[..]).unwrap().unpublished, vec![0]);
    }
    #[test] fn a_field_masked_where_the_land_fraction_is_masked_remaps() {
        let out=apply(&masked_mesh(),&[f64::NAN,1.,2.,3.,4.,5.],Method::Idw4).unwrap();
        assert!(out.iter().all(|x| x.is_finite() && (1.0..=5.0).contains(x)));
    }
    #[test] fn a_field_that_publishes_an_unpublished_cell_is_refused() {
        assert!(apply(&masked_mesh(),&[9.,1.,2.,3.,4.,5.],Method::Idw4).is_err());
    }
    #[test] fn a_plan_naming_an_unpublished_donor_is_invalid() {
        let mut p=masked_mesh(); p.unpublished=vec![p.stencils[0].ids[0]];
        assert!(p.validate().is_err());
    }
    #[test] fn fewer_than_four_published_cells_or_a_bad_fraction_are_refused() {
        let (lat,lon)=([0.,0.,0.125,0.125],[0.,0.125,0.,0.125]);
        assert!(build_plan(&lat,&lon,&[f64::NAN,1.,1.,1.],native_grid(4),78,target(),80_000.).is_err());
        assert!(build_plan(&lat,&lon,&[1.5,1.,1.,1.],native_grid(4),78,target(),80_000.).is_err());
        assert!(build_plan(&lat,&lon,&[f64::INFINITY,1.,1.,1.],native_grid(4),78,target(),80_000.).is_err());
    }
    #[test] fn a_previous_plan_version_is_refused() {
        let mut bytes=Vec::new(); plan().write(&mut bytes).unwrap();
        bytes[..8].copy_from_slice(b"GDT101P2");
        assert!(Plan::read(&bytes[..]).is_err());
    }
    #[test] fn holes_in_geometry_are_not_extrapolated() {
        assert!(build_plan(&[30.,30.,30.125,30.125], &[0.,0.125,0.,0.125],&[1.;4],native_grid(4),78,target(),80_000.).is_err());
    }
    #[test] fn kd_matches_bruteforce_including_ties() {
        let mut points=Vec::new();for n in 0..1000 {points.push(Point{xyz:xyz((n%53) as f64-26.,(n%101) as f64-50.).unwrap(),id:n});}
        let original=points.clone();partition(&mut points,0);
        for (lat,lon) in [(0.,0.),(3.14,8.7),(-22.,-49.),(25.,45.)] {
            let q=xyz(lat,lon).unwrap();let mut kd=[EMPTY;4];let mut brute=[EMPTY;4];
            search(&points,&q,0,&mut kd);for p in &original {insert(&mut brute,Candidate{d2:distance2(&q,&p.xyz),id:p.id});}
            assert_eq!(kd.map(|p|p.id),brute.map(|p|p.id));
        }
    }
    #[test] fn wire_preserves_product_and_uses_northward_rows() {
        let original=template();let output=encode_regular(&original,target(),&[1.,2.,3.,4.]).unwrap();
        let a=envelope(&original).unwrap();let b=envelope(&output).unwrap();
        assert_eq!(a.sections[1],b.sections[1]);assert_eq!(a.sections[4],b.sections[4]);
        assert_eq!(b.sections[3].unwrap()[71],0x40);assert_eq!(b.sections[5].unwrap()[9..12],[0,4,1]);
    }
    #[test] fn wire_missing_bitmap_and_empty_simple_field() {
        let bytes=encode_regular(&template(),target(),&[1.,f64::NAN,2.,f64::NAN]).unwrap();
        let e=envelope(&bytes).unwrap();assert_eq!(e.sections[6].unwrap()[6],0b10100000);
        assert_eq!(e.sections[7].unwrap().len(),13);
        let empty=encode_regular(&template(),target(),&[f64::NAN;4]).unwrap();
        let e=envelope(&empty).unwrap();assert_eq!(e.sections[5].unwrap().len(),21);assert_eq!(e.sections[7].unwrap().len(),5);
    }
    #[test] fn wire_rejects_infinite_values_and_extra_envelopes() {
        assert!(encode_regular(&template(),target(),&[f64::INFINITY;4]).is_err());
        let mut data=template();data.extend(template());assert!(envelope(&data).is_err());
    }
    #[test] fn wire_negative_latitude_is_sign_magnitude() {
        let t=Target{south:-30.,west:179.875,..target()};let s=regular_grid(t).unwrap();
        assert_eq!(u32::from_be_bytes(s[46..50].try_into().unwrap()),0x80000000|30_000_000);
        assert_eq!(u32::from_be_bytes(s[59..63].try_into().unwrap()),180_000_000);
    }
}
